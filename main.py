import argparse, glob, os, torch, warnings, time
from tools import *
from trainer import *
from dataLoader import *


parser = argparse.ArgumentParser(description = "AV speaker recognition on VoxCeleb2")

### Training setting
parser.add_argument('--max_epoch',  type=int,   default=100,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=100,      help='Batch size')
parser.add_argument('--frame_len',  type=int,   default=100,      help='Input length of utterance')
parser.add_argument('--n_cpu',      type=int,   default=12,       help='Number of loader threads')
parser.add_argument('--n_class',    type=int,   default=5994,     help='Number of speakers')
parser.add_argument('--test_step',  type=int,   default=1,        help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,    help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.65,     help='Learning rate decay every [test_step] epochs')
parser.add_argument("--lr_decay_start",   type=float, default=1,     help='Learning rate decay every [test_step] epochs')
parser.add_argument("--lr_decay_every",   type=float, default=1,     help='Learning rate decay every [test_step] epochs')
parser.add_argument("--lr_decay_rate",   type=float, default=0.65,     help='Learning rate decay every [test_step] epochs')

### Data path
parser.add_argument('--train_list', type=str,   default="/data08/VoxCeleb2/train_all.txt",     help='The path of the training list')
parser.add_argument('--train_path', type=str,   default="/data08/VoxCeleb2/",                  help='The path of the training data')
parser.add_argument('--eval_trials',type=str,   default="/data08/VoxCeleb1/O_trials.txt",      help='The path of the evaluation trials, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--eval_list',  type=str,   default="/data08/VoxCeleb1/O_list.txt",        help='The path of the evaluation list, contains all evaluation data info')
parser.add_argument('--eval_path',  type=str,   default="/data08/VoxCeleb1/",                  help='The path of the evaluation data')
parser.add_argument('--musan_path',   type=str, default="/misc/scratch02/reco/Corpora/VoxCeleb_August2021/musan_split/")
parser.add_argument('--rir_path',     type=str, default="/misc/scratch02/reco/Corpora/VoxCeleb_August2021/RIRS_NOISES/simulated_rirs/")
parser.add_argument('--num_eval_frames', type=int,   default=1,       help='Number of eval frames, higher will be better')
parser.add_argument('--save_path',  type=str,    default="", help='Path to save the clean list')

### Initial modal path
parser.add_argument('--initial_model_a',  type=str,   default="",  help='Path of the initial_model, audio')
parser.add_argument('--initial_model_v',  type=str,   default="",  help='Path of the initial_model, visual')
parser.add_argument('--model_av',  type=str,   default="", help='Path of the model, audiovisual_model')
parser.add_argument('--eval_data',  type=str,   default="", help='Path of the eval data partition, data')

### Model & loss setting
parser.add_argument('--margin_a',       type=float, default=0.2,    help='Loss margin for audio training')
parser.add_argument('--scale_a',        type=float, default=30,     help='Loss scale for audio training')
parser.add_argument('--model_a',        type=str,   default="ecapa1024",     help='ecapa512 or ecapa1024')

parser.add_argument('--margin_v',       type=float, default=0.4,    help='Loss margin for visual training')
parser.add_argument('--scale_v',        type=float, default=64,     help='Loss scale for visual training')
parser.add_argument('--model_v',        type=str,   default="res18",     help='resnet 18 or 50')

###  Others
parser.add_argument('--train',   dest='train', action='store_true', help='Do training')
parser.add_argument('--eval',    dest='eval', action='store_true', help='Do evaluation')

## Init folders
args = init_system(parser.parse_args())
## Init loader
args = init_loader(args)
## Init trainer
s = init_trainer(args)

## Evaluate only
if args.eval == True:
	EER, minDCF = s.eval_network(args)
	quit()

	
#model_save_path_a = os.path.join(args.save_path, 'model_a')
#model_save_path_v = os.path.join(args.save_path, 'model_v')
model_save_path_av = os.path.join(args.save_path, 'model_av')


## Train only
if args.train == True:
	while args.epoch < args.max_epoch:
		s.train_network(args)
		if args.epoch % args.test_step == 0:
			EER, minDCF = s.eval_network(args)
			s.save_parameters(model_save_path_av, "model_%04d.model"%args.epoch, args.epoch, EER, minDCF, 'audiovisual')
		args.epoch += 1
	quit()
