import torch, sys, os, tqdm, numpy, soundfile, time, pickle, cv2, glob, random, scipy
import torch.nn as nn
from tools import *
from loss import *
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import defaultdict, OrderedDict
from torch.cuda.amp import autocast,GradScaler
from typing import Optional, Any, Dict
import fsspec
from dataclasses import dataclass, asdict

@dataclass
class LossConfig:
    margin_v: float = 0.4
    margin_a: float = 0.2
    scale_a: int = 30
    scale_v: int = 64
    n_class: int = 1150


@dataclass
class OptimizerConfig:
	weight_decay: float = 0.1
	learning_rate: float = 0.001
	learning_rate_decay: float = 0.65
	lr_decay_start: int = 5
	lr_decay_every: int = 2
	lr_decay_rate: float = 0.8
	weight_decay: float = 2e-5

@dataclass
class TrainerConfig:
	max_epochs: int=200
	batch_size: int=200
	frame_len: int=100
	n_class: int=1150
	data_loader_workers: int=12
	grad_norm_clip: float=1.0
	snapshot_path: str='exps/debug/snapshot/snapshot.pt'
	save_every: int=3
	score_file: str = 'exps/debug/score.txt'
	eval_trials: str = '../../veri_val_face.txt'

@dataclass
class Snapshot:
	model_state: 'OrderedDict[str, torch.Tensor]'
	loss_state: 'OrderedDict[str, torch.Tensor]'
	optimizer_state: Dict[str, Any]
	finished_epoch: int
	

def set_lr(optimizer, lr):
	for group in optimizer.param_groups:
		group['lr'] = lr

class Trainer(nn.Module):
	def __init__(self, a_model, v_model, av_model, optimizer_cfg: OptimizerConfig, loss_cfg:LossConfig, local_rank):
		super(Trainer, self).__init__()

		#self.aud_model = a_model.to(local_rank)
		#self.aud_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.aud_model) 
		#self.vis_model = v_model.to(local_rank)
		#self.vis_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.vis_model)
		#self.fusion_model = av_model.to(local_rank)
		#self.fusion_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.fusion_model) 
		self.speaker_face_loss    = AAMsoftmax(n_class = loss_cfg.n_class, m = loss_cfg.margin_v, s = loss_cfg.scale_v, c = 512).to(local_rank)
		#self.speaker_face_loss = self.speaker_face_loss.to(local_rank)
		#self.speaker_face_loss = DDP(self.speaker_face_loss, device_ids=[local_rank])

		self.fusion_model = av_model
		self.aud_model = a_model
		self.vis_model = v_model

		#self.fusion_model = DDP(self.fusion_model, device_ids=[local_rank], find_unused_parameters=True)
		#self.aud_model = DDP(self.aud_model, device_ids=[local_rank], find_unused_parameters=True)
		#self.vis_model = DDP(self.vis_model, device_ids=[local_rank], find_unused_parameters=True)
		
		#self.optim           = torch.optim.Adam(list(self.fusion_model.state_dict().items()) + list(self.speaker_face_loss.state_dict().items()), lr = optimizer_cfg.learning_rate, weight_decay = optimizer_cfg.weight_decay)
		self.optim           = torch.optim.Adam(self.parameters(), lr = optimizer_cfg.learning_rate, weight_decay = optimizer_cfg.weight_decay)
		#self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma = 0.65)
		#print(" Speech model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1e6))
		#print(" Face model para number = %.2f"%(sum(param.numel() for param in self.face_encoder.parameters()) / 1e6))
		#self.fusion_model = CAM().cuda()

	def _load_snapshot(self, trainerconfig):
		try:
			snapshot = fsspec.open(trainerconfig.snapshot_path)
			with snapshot as f:
				snapshot_data = torch.load(f, map_location="cpu")
		except FileNotFoundError:
			print("Snapshot not found. Training model from scratch")
			return 

		snapshot = Snapshot(**snapshot_data)
		self.fusion_model.load_state_dict(snapshot.model_state)
		self.optim.load_state_dict(snapshot.optimizer_state)
		self.speaker_face_loss.load_state_dict(snapshot.loss_state)
		self.epochs_run = snapshot.finished_epoch
		print(f"Resuming training from snapshot at Epoch {self.epochs_run}")


	def _save_snapshot(self,epoch, trainerconfig):
		# capture snapshot
		model = self.fusion_model
		raw_model = model.module if hasattr(model, "module") else model
		snapshot = Snapshot(
			model_state=raw_model.state_dict(),
			optimizer_state=self.optim.state_dict(),
			loss_state = self.speaker_face_loss.state_dict(),
			finished_epoch=epoch
		)
		# save snapshot
		snapshot = asdict(snapshot)
		torch.save(snapshot, trainerconfig.snapshot_path)
			
		print(f"Snapshot saved at epoch {epoch}")

	def train_network(self, epoch, trainloader, optimizercfg, trainercfg):
		#self.fusion_model.train()
		#self.aud_model.eval()
		#self.vis_model.eval()
		self.train()
		scaler = GradScaler()
		#self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0

		if epoch > optimizercfg.lr_decay_start and optimizercfg.lr_decay_start >= 0:
			frac = (epoch - optimizercfg.lr_decay_start) // optimizercfg.lr_decay_every
			decay_factor = optimizercfg.lr_decay_rate ** frac
			lr = optimizercfg.learning_rate * decay_factor
			set_lr(self.optim, lr)  # set the decayed rate
		else:
			lr = optimizercfg.learning_rate
		#lr = self.optim.param_groups[0]['lr']
		epoch_loss = 0
		n = 0
		acc = 0
		time_start = time.time()


		for num, (speech, face, labels) in enumerate(trainloader, start = 1):

			#if num>3:
			#	break
			self.optim.zero_grad(set_to_none=True)
			labels      = torch.LongTensor(labels)	
			face        = face.div_(255).sub_(0.5).div_(0.5)
			with autocast():
				with torch.no_grad():
					aud_embeddings = self.aud_model(speech)
					vis_embeddings = self.vis_model(face)
				#_, AVloss = self.fusion_model(aud_embeddings, vis_embeddings, labels, train=True)
				AV_embeddings = self.fusion_model(aud_embeddings, vis_embeddings)

				AVloss, _ = self.speaker_face_loss.forward(AV_embeddings, labels.to(AV_embeddings.get_device()))
			scaler.scale(AVloss).backward()
			scaler.step(self.optim)
			scaler.update()

			index += len(labels)
			loss += (AVloss).detach().cpu().numpy()

			time_used = time.time() - time_start

			sys.stderr.write(" [%2d] %.2f%% (est %.1f mins) Lr: %5f, Loss: %.5f\r"%\
			(epoch, 100 * (num / trainloader.__len__()), time_used * trainloader.__len__() / num / 60, lr, loss/(num)))
			sys.stderr.flush()
		sys.stdout.write("\n")
		with open(trainercfg.score_file, 'w') as score_file:
			score_file.write("%d epoch, LR %f, LOSS %f\n"%(epoch, lr, loss/num))
			score_file.flush()
		return
		
	#@torch.no_grad()
	def eval_network(self, epoch, evalloader, trainerconfig):
		#self.fusion_model.eval()
		#self.aud_model.eval()
		#self.vis_model.eval()
		self.eval()
		scores_av, scores_a, scores_v, labels, res = [], [], [], [], []
		embeddings = {}
		lines = open(trainerconfig.eval_trials).read().splitlines()
		#print(len(evalloader))
		#sys.exit()

		for a_data, v_data, filenames in tqdm.tqdm(evalloader, total = len(evalloader)):
			#print(a_data.shape)
			#print(v_data.shape)
			#print(filenames)
			#sys.exit()
			with torch.no_grad():
				a_data = a_data.squeeze(0)
				v_data = v_data.squeeze(0).squeeze(2)
				#AV_embeddings, _ = self.model(a_data, v_data)
				aud_embeddings = self.aud_model(a_data)
				vis_embeddings = self.vis_model(v_data)
				#AV_embeddings = self.fusion_model(aud_embeddings, vis_embeddings, labels, train=False)
				AV_embeddings = self.fusion_model(aud_embeddings, vis_embeddings)

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
		with open(trainerconfig.score_file, 'w') as score_file:
			score_file.write("EER_a %2.4f, min_a %.4f\n"%(res[0], res[1]))
			score_file.flush()
		return
