#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=15000M
#SBATCH --job-name=RJCAFusion
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=gnana-praveen.rajasekhar@crim.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out
export PATH="/misc/home/reco/rajasegp/anaconda3/bin:$PATH"
#source activate py36_ssd20
source activate AVfusion
python main.py \
--train_list txt_files/train_list_face_new.txt \
--train_path /misc/scratch11/Voxceleb1_Data/Voxceleb1_Data/Insightface_Aligned_Images/ \
--eval_trials txt_files/veri_val_face.txt \
--eval_list txt_files/val_list_face_new.txt \
--eval_path /misc/scratch11/Voxceleb1_Data/Voxceleb1_Data/Insightface_Aligned_Images/ \
--initial_model_a pretrainedmodels/audio_model.model \
--initial_model_v pretrainedmodels/video_model.model \
--save_path exps/debug \
--n_class 1150 \
--lr 0.001 \
--lr_decay 0.65 \
--scale_a 30 \
--margin_a 0.2 \
--model_a ecapa1024 \
--scale_v 64 \
--margin_v 0.4 \
--model_v res18 \
--test_step 1 \
--max_epoch 200 \
--train
