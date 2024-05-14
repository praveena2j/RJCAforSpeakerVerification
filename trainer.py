import torch, sys, os, tqdm, numpy, soundfile, time, pickle, cv2, glob, random, scipy
import torch.nn as nn
from tools import *
from loss import *
from audiomodel import *
from visualmodel import *
from collections import defaultdict, OrderedDict
from orig_cam import CAM
from ASP import Attentive_Statistics_Pooling
from torch.cuda.amp import autocast,GradScaler

def init_trainer(args):
	s = trainer(args)
	args.epoch = 1
	if args.initial_model_a != '':
		print("Model %s loaded from previous state!"%(args.initial_model_a))
		s.load_parameters(args.initial_model_a, 'A')
	elif len(args.modelfiles_a) >= 1:
		print("Model %s loaded from previous state!"%args.modelfiles_a[-1])
		args.epoch = int(os.path.splitext(os.path.basename(args.modelfiles_a[-1]))[0][6:]) + 1
		s.load_parameters(args.modelfiles_a[-1], 'A')

	if args.initial_model_v != '':
		print("Model %s loaded from previous state!"%(args.initial_model_v))
		s.load_parameters(args.initial_model_v, 'V')
	elif len(args.modelfiles_v) >= 1:
		print("Model %s loaded from previous state!"%args.modelfiles_v[-1])
		args.epoch = int(os.path.splitext(os.path.basename(args.modelfiles_v[-1]))[0][6:]) + 1
		s.load_parameters(args.modelfiles_v[-1], 'V')

	if args.eval == True:
		print("Model %s loaded from input state!"%(args.model_av))
		s.fusion_model.load_state_dict(torch.load(args.model_av)['net'])
		print(torch.load(args.model_av)['EER'])
		print(torch.load(args.model_av)['minDCF'])
		print(torch.load(args.model_av)['epoch'])
	return s


def set_lr(optimizer, lr):
	for group in optimizer.param_groups:
		group['lr'] = lr

class FusionModel(nn.Module):
	def __init__(self):
		super(FusionModel, self).__init__()
		self.JCA_model = CAM().cuda() 
		self.asp = Attentive_Statistics_Pooling(704).cuda()
		self.linear = nn.Linear(1408, 512).cuda()


	def forward(self, a_feat, v_feat):
		AV_feats = self.JCA_model(a_feat, v_feat)
		AV_feats = self.asp(AV_feats)
		AV_embeddings = self.linear(AV_feats)
		return AV_embeddings

class trainer(nn.Module):
	def __init__(self, args):
		super(trainer, self).__init__()
		self.speaker_encoder = ECAPA_TDNN(model = args.model_a).cuda()
		self.speaker_loss    = AAMsoftmax(n_class = args.n_class, m = args.margin_a, s = args.scale_a, c = 192).cuda()	
		self.speaker_face_loss    = AAMsoftmax(n_class = args.n_class, m = args.margin_v, s = args.scale_v, c = 512).cuda()	
		self.face_encoder    = IResNet(model = args.model_v).cuda()
		self.fusion_model =  FusionModel().cuda()

		# self.fusion_model = CAM().cuda() 
		##self.prediction = nn.Sequential(
		#		   	Attentive_Statistics_Pooling(704).cuda(),
		#			nn.Linear(1408, 512).cuda() 
		#			)

		#self.asp = Attentive_Statistics_Pooling(704).cuda()
		#self.asp = Attentive_Statistics_Pooling().cuda()
		#self.linear = nn.Linear(1408, 512).cuda()
		#self.loss = nn.MultiLabelSoftMarginLoss().cuda()
		self.face_loss       = AAMsoftmax(n_class =  1150, m = args.margin_v, s = args.scale_v, c = 512).cuda()
		self.optim           = torch.optim.Adam(self.parameters(), lr = args.lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = args.test_step, gamma = args.lr_decay)
		print(" Speech model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1e6))
		print(" Face model para number = %.2f"%(sum(param.numel() for param in self.face_encoder.parameters()) / 1e6))
		#self.fusion_model = CAM().cuda()

	def train_network(self, args):
		self.speaker_encoder.eval()
		self.speaker_loss.eval()
		self.face_encoder.eval()
		self.face_loss.eval()
		self.fusion_model.train()
		self.speaker_face_loss.train()

		scaler = GradScaler()
		self.scheduler.step(args.epoch - 1)
		index, top1, loss = 0, 0, 0

		if args.epoch > args.lr_decay_start and args.lr_decay_start >= 0:
			frac = (args.epoch - args.lr_decay_start) // args.lr_decay_every
			decay_factor = args.lr_decay_rate ** frac
			lr = args.lr * decay_factor
			set_lr(self.optim, lr)  # set the decayed rate
		else:
			lr = args.lr
		#lr = self.optim.param_groups[0]['lr']
		epoch_loss = 0
		n = 0
		acc = 0
		time_start = time.time()

		for num, (speech, face, labels) in enumerate(args.trainLoader, start = 1):
			self.zero_grad()

			labels      = torch.LongTensor(labels).cuda()

			face        = face.div_(255).sub_(0.5).div_(0.5)
			#if num> 5:
			#	break
			with autocast():
				with torch.no_grad():
					b, seq, ch, h, w = face.size()
					speech = speech.view(b*seq, -1)
					face = face.view(b*seq, ch, h, w)
					a_embedding   = self.speaker_encoder.forward(speech.cuda(), aug = True)	
					v_embedding   = self.face_encoder.forward(face.cuda())	
					a_embedding = a_embedding.view(b, seq, -1)
					v_embedding = v_embedding.view(b, seq, -1)
					#visual_feats = torch.empty((speech.shape[0], speech.shape[1], 512), dtype=face.dtype).cuda()
					#aud_feats = torch.empty((speech.shape[0], speech.shape[1], 192), dtype=speech.dtype).cuda()
					
					#for i in range(face.shape[0]):
					#	a_embedding   = self.speaker_encoder.forward(speech[i, :, :].cuda(), aug = True)	
					#	v_embedding   = self.face_encoder.forward(face[i, :,:,:,:].cuda())	
			
					#	visual_feats[i,:,:] = v_embedding
					#	aud_feats[i,:,:] = a_embedding
				
				AV_embeddings = self.fusion_model(a_embedding, v_embedding)
				#AV_embeddings = self.linear(AV_feats)
				#AV_feats = self.asp(AVfeatures)

				#AV_embeddings = self.linear(AV_feats)
				#AV_embeddings = self.prediction(AV_feats)
				
				#loss = self.loss(score, labels)
				AVloss, _ = self.speaker_face_loss.forward(AV_embeddings, labels)			
			scaler.scale(AVloss).backward()
			scaler.step(self.optim)
			scaler.update()

			index += len(labels)
			loss += (AVloss).detach().cpu().numpy()

			time_used = time.time() - time_start
			#print("Training takes " + str(time_used))
			sys.stderr.write(" [%2d] %.2f%% (est %.1f mins) Lr: %5f, Loss: %.5f\r"%\
			(args.epoch, 100 * (num / args.trainLoader.__len__()), time_used * args.trainLoader.__len__() / num / 60, lr, loss/(num)))
			sys.stderr.flush()
		sys.stdout.write("\n")

		args.score_file.write("%d epoch, LR %f, LOSS %f\n"%(args.epoch, lr, loss/num))
		args.score_file.flush()
		return
		
	def eval_network(self, args):
		self.eval()
		scores_av, scores_a, scores_v, labels, res = [], [], [], [], []
		embeddings = {}
		lines = open(args.eval_trials).read().splitlines()
		for a_data, v_data, filenames in tqdm.tqdm(args.evalLoader, total = len(args.evalLoader)):
			with torch.no_grad():

				visual_feats = torch.empty((a_data.shape[1], a_data.shape[2], 512), dtype=v_data.dtype).cuda()
				aud_feats = torch.empty((a_data.shape[1], a_data.shape[2], 192), dtype=a_data.dtype).cuda()
				for i in range(v_data.shape[1]):
					a_data = a_data.squeeze(0)
					v_data = v_data.squeeze(0).squeeze(2)
					a_embedding   = self.speaker_encoder.forward(a_data[i, :, :].cuda())
					v_embedding   = self.face_encoder.forward(v_data[i, :,:,:,:].cuda())	
					visual_feats[i,:,:] = v_embedding
					aud_feats[i,:,:] = a_embedding
				
				AV_embeddings = self.fusion_model(aud_feats, visual_feats)
				
				#AV_embeddings = self.fusion_model(a_embedding, v_embedding)

				#AV_feats = self.asp(AVfeatures)
				#AV_embeddings = self.linear(AV_feats)
				for num in range(len(filenames)):
					filename = filenames[num][0]
					AVembeddings = torch.unsqueeze(AV_embeddings[num, :], dim = 0)
					embeddings[filename] = F.normalize(AVembeddings, p=2, dim=1)

		for line in tqdm.tqdm(lines):			
			a1 = embeddings[line.split()[1]]
			a2 = embeddings[line.split()[2]]
			score_a = torch.mean(torch.matmul(a1, a2.T)).detach().cpu().numpy()
			scores_a.append(score_a)
			labels.append(int(line.split()[0]))

		for score in [scores_a]:
			EER = tuneThresholdfromScore(score, labels, [1, 0.1])[1]
			fnrs, fprs, thresholds = ComputeErrorRates(score, labels)
			minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
			res.extend([EER, minDCF])
		
		print('EER_a %2.4f, min_a %.4f\n'%(res[0], res[1]))
		args.score_file.write("EER_a %2.4f, min_a %.4f\n"%(res[0], res[1]))
		args.score_file.flush()
		return res[0], res[1]

	def save_parameters(self, path, model_save_name, epoch,EER, minDCF, modality):
		#if modality == 'A':	
		#	model = OrderedDict(list(self.speaker_encoder.state_dict().items()) + list(self.speaker_loss.state_dict().items()))
		#	torch.save(model, os.path.join(path, model_save_name))
		#if modality == 'V':
		#	model = OrderedDict(list(self.face_encoder.state_dict().items()) + list(self.face_loss.state_dict().items()))
		#	torch.save(model, os.path.join(path, model_save_name))
		if modality == 'audiovisual':		
			state = {
					'net': self.fusion_model.state_dict(),
					'EER': EER,
					'minDCF' : minDCF,
					'epoch': epoch,
			}
			torch.save(state, os.path.join(path, model_save_name))

	def load_parameters(self, path, modality):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			if modality == 'A':
				if ('face_encoder.' not in name) and ('face_loss.' not in name):
					if ('fusion_model.' not in name) and ('speaker_face_loss.' not in name):
						if ('speaker_encoder.' not in name) and ('speaker_loss.' not in name):
							if name == 'weight':
								name = 'speaker_loss.' + name
							else:
								name = 'speaker_encoder.' + name
						self_state[name].copy_(param)
			if modality == 'V':				
				if ('speaker_encoder.' not in name) and ('speaker_loss.' not in name):
					if ('fusion_model.' not in name) and ('speaker_face_loss.' not in name):					
						if ('face_encoder.' not in name) and ('face_loss.' not in name):
							if name == 'weight':
								name = 'face_loss.' + name
							else:
								name = 'face_encoder.' + name
						self_state[name].copy_(param)