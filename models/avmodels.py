import torch.nn as nn
from dataclasses import dataclass
from .audiomodel import ECAPA_TDNN
from loss import AAMsoftmax
from .visualmodel import IResNet
from .orig_cam import CAM
from .ASP import Attentive_Statistics_Pooling
import torch

@dataclass
class ModelConfig:
    # model configurations
    initial_model_a: str = '../speaker/exps/debug/model_a/model_0199.model'
    initial_model_v: str = '../face/exps/debug/model_v/model_0094.model'
    model_a: str = 'ecapa1024'
    model_v: str = 'res18'

@dataclass
class LossConfig:
    margin_v: float = 0.4
    margin_a: float = 0.2
    scale_a: int = 30
    scale_v: int = 64
    n_class: int = 1150


def init_AV_models(model_cfg, loss_cfg):
	audio_model = AudioModel(model_cfg, loss_cfg)
	visual_model = VisualModel(model_cfg, loss_cfg)
	print("AudioModel %s loaded from previous state!"%(model_cfg.initial_model_a))
	audio_model.load_audio_parameters(model_cfg.initial_model_a)
	
	print("VisualModel %s loaded from previous state!"%(model_cfg.initial_model_v))
	visual_model.load_visual_parameters(model_cfg.initial_model_v)
	return audio_model, visual_model


class AudioModel(nn.Module):
	def __init__(self, model_cfg: ModelConfig, loss_cfg:LossConfig):
		super(AudioModel, self).__init__()
		self.speaker_encoder = ECAPA_TDNN(model = model_cfg.model_a)
		self.speaker_loss    = AAMsoftmax(n_class = loss_cfg.n_class, m = loss_cfg.margin_a, s = loss_cfg.scale_a, c = 192)	
	
	def forward(self, speech):
		#with torch.no_grad():
		b, seq, _ = speech.size()
		speech = speech.view(b*seq, -1)

		a_embedding   = self.speaker_encoder.forward(speech, aug = True)	
		a_embedding = a_embedding.view(b, seq, -1)

		return a_embedding
	def load_audio_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			if ('face_encoder.' not in name) and ('face_loss.' not in name):
				if ('speaker_encoder.' not in name) and ('speaker_loss.' not in name):
					if name == 'weight':
						name = 'speaker_loss.' + name
					else:
						name = 'speaker_encoder.' + name
				self_state[name].copy_(param)
			
class VisualModel(nn.Module):
	def __init__(self, model_cfg: ModelConfig, loss_cfg:LossConfig):
		super(VisualModel, self).__init__()
		self.face_encoder    = IResNet(model = model_cfg.model_v)
		self.face_loss       = AAMsoftmax(n_class =  loss_cfg.n_class, m = loss_cfg.margin_v, s = loss_cfg.scale_v, c = 512)
		
	def forward(self, face):
		b, seq, ch, h, w = face.size()
		face = face.view(b*seq, ch, h, w)

		v_embedding   = self.face_encoder.forward(face)	
		v_embedding = v_embedding.view(b, seq, -1)

		return v_embedding

	def load_visual_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():	
			if ('speaker_encoder.' not in name) and ('speaker_loss.' not in name):					
				if ('face_encoder.' not in name) and ('face_loss.' not in name):
					if name == 'weight':
						name = 'face_loss.' + name
					else:
						name = 'face_encoder.' + name
				self_state[name].copy_(param)



class FusionModel(nn.Module):
	def __init__(self, model_cfg: ModelConfig, loss_cfg:LossConfig):
		super(FusionModel, self).__init__()
		self.JCA_model = CAM()
		self.asp = Attentive_Statistics_Pooling(704)
		self.linear = nn.Linear(1408, 512)


	def forward(self, a_embedding, v_embedding):

		#with torch.no_grad():
		AV_feats = self.JCA_model(a_embedding, v_embedding)
		AV_feats = self.asp(AV_feats)
		AV_embeddings = self.linear(AV_feats)
		#if train:
		#	AVloss, _ = self.speaker_face_loss.forward(AV_embeddings, labels)
		#	return AV_embeddings, AVloss
		#else:
		return AV_embeddings


