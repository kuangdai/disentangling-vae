#!/bin/bash
#SBATCH -p pearl
#SBATCH -n 4
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=20G
#SBATCH -t 02:00:00
#SBATCH --job-name=bvae

export OMP_NUM_THREADS=4

module load OpenMPI/4.1.0-GCC-9.3.0
mpirun --mca btl_openib_allow_ib 1 --mca btl_openib_if_include mlx5_0:1 singularity exec --nv ../../../torch.simg python run.py
