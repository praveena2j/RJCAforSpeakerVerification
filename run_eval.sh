#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=15000M
#SBATCH --gres=gpu:1
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
--train_list ../face/train_list_face_new.txt \
--train_path /lu/bf_scratch/patx/VoxCeleb1_faces/unzippedIntervalFaces/data/ \
--eval_trials ../../veri_val_face.txt \
--eval_list ../face/val_list_face_new.txt \
--eval_path /lu/bf_scratch/patx/VoxCeleb1_faces/unzippedIntervalFaces/data/ \
--save_path exps/debug \
--n_class 1150 \
--initial_model_a ../speaker/exps/debug/model_a/model_0199.model \
--initial_model_v ../face/exps/debug/model_v/model_0094.model \
--model_a ecapa1024 \
--model_v res18 \
--eval \