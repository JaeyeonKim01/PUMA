#!/bin/bash
#SBATCH --job-name=puma_tinygsm
#SBATCH --account=<your-account-here>
#SBATCH --partition=<your-partition-here>
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=360GB
#SBATCH --time=3-00:00:00

conda deactivate
conda activate puma
module load cuda # if needed

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(srun -N1 -n1 -w "$MASTER_HOST" hostname -I | awk '{print $1}')
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --gpus-per-task=4 \
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    train.py --cfg yaml_files/tinygsm_puma.yaml
