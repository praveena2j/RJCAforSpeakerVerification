#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=DDP-RJCAFusion
#SBATCH --time=5-00:00:00
#SBATCH --mail-user=gnana-praveen.rajasekhar@crim.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --error=err/%x.%j.err
#SBATCH --output=out/%x.%j.out

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

export PATH="/misc/home/reco/rajasegp/anaconda3/bin:$PATH"
#source activate py36_ssd20
source activate AVfusion

#export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

srun torchrun \
--nnodes 1 \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint 127.0.0.1:29600 \
main.py
