import argparse, glob, os, torch, warnings, time
from tools import *
from trainer import *
from dataLoader import *
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from omegaconf import DictConfig
from torch.distributed import init_process_group, destroy_process_group
from models.avmodels import ModelConfig, LossConfig
from trainer import OptimizerConfig
from dataLoader import DataConfig
from trainer import Trainer
from models.avmodels import AudioModel, VisualModel, FusionModel
from models.avmodels import init_AV_models
import sys
import numpy as np
from socket import gethostname
import hydra

def ddp_setup():
	init_process_group(backend="nccl")

#def set_random_seeds(random_seed=0):
#    torch.manual_seed(random_seed)
#    torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.benchmark = False
#    np.random.seed(random_seed)
#    random.seed(random_seed)


def get_train_objs(model_cfg: ModelConfig, optimizer_cfg: OptimizerConfig, data_cfg: DataConfig, loss_cfg:LossConfig):
	trainloader, valloader = init_loader(data_cfg)
	a_model, v_model = init_AV_models(model_cfg, loss_cfg)
	#a_model = AudioModel(model_cfg, loss_cfg)
	#v_model = VisualModel(model_cfg, loss_cfg)
	av_model = FusionModel(model_cfg, loss_cfg)
	#optimizer = create_optimizer(model, opt_cfg)
	#optimizer    = torch.optim.Adam(model.parameters(), lr = optimizer_cfg.lr, optimizer_cfg = 2e-5)
	return a_model, v_model, av_model, trainloader, valloader

@hydra.main(version_base=None, config_path=".", config_name="train_cfg")
def main(cfg: DictConfig):

	## DDP config
	ddp_setup()

	model_cfg = ModelConfig(**cfg['model_config'])
	loss_cfg = LossConfig(**cfg['loss_config'])
	optimizer_cfg = OptimizerConfig(**cfg['optimizer_config'])
	data_cfg = DataConfig(**cfg['data_config'])
	trainer_cfg = TrainerConfig(**cfg['trainer_config'])
	print(data_cfg.batch_size)
	print(optimizer_cfg.learning_rate)
	print(data_cfg.data_loader_workers)
	a_model, v_model, av_model, trainloader, valloader = get_train_objs(model_cfg, optimizer_cfg, data_cfg, loss_cfg)

	local_rank = int(os.environ["LOCAL_RANK"])
	global_rank = int(os.environ["RANK"])  
	
	save_every = trainer_cfg.save_every
	
	aud_model = a_model.to(local_rank)
	aud_model = DDP(aud_model, device_ids=[local_rank], find_unused_parameters=True)
	aud_model = nn.SyncBatchNorm.convert_sync_batchnorm(aud_model) 
	vis_model = v_model.to(local_rank)
	vis_model = DDP(vis_model, device_ids=[local_rank], find_unused_parameters=True)
	vis_model = nn.SyncBatchNorm.convert_sync_batchnorm(vis_model)
	fusion_model = av_model.to(local_rank)
	fusion_model = DDP(fusion_model, device_ids=[local_rank], find_unused_parameters=True)

	trainer = Trainer(aud_model, vis_model, fusion_model, optimizer_cfg, loss_cfg, local_rank)
	if trainer_cfg.snapshot_path is None:
		trainer_cfg.snapshot_path = "snapshot.pt"
	#trainer._load_snapshot(trainer_cfg)
	epochs_run = 0
	for epoch in range(epochs_run, trainer_cfg.max_epochs):
		epoch += 1
		time_start = time.time()
		trainer.train_network(epoch, trainloader, optimizer_cfg, trainer_cfg)
		time_used = time.time() - time_start
		print("Training Time takes:"+ str(time_used))
		print("Training completed")
		if local_rank == 0 and epoch % save_every == 0:
			trainer._save_snapshot(epoch, trainer_cfg)
		# eval run
		trainer.eval_network(epoch, valloader, trainer_cfg)
		print("Validation completed")
	
	destroy_process_group()
	#else:
	#	s = WrappedModel(s).cuda(args.gpu)


if __name__ == '__main__':
	main()