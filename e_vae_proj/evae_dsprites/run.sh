#!/bin/bash
#SBATCH -p pearl
#SBATCH --job-name=bvae
#SBATCH --mem-per-cpu=32G
#SBATCH --mem-per-gpu=32G
#SBATCH --threads-per-core=1
#SBATCH -n 8
#SBATCH --gres=gpu:8
#SBATCH -t 5-00:00:00

module load OpenMPI/4.1.0-GCC-9.3.0
mpirun --mca btl_openib_allow_ib 1 --mca btl_openib_if_include mlx5_0:1 singularity exec --nv ../../../torch.simg python run.py
