#!/bin/bash
#SBATCH -p pearl
#SBATCH --job-name=bvae
#SBATCH --time=4-00:00:00
#SBATCH -n 12
#SBATCH --gres=gpu:12
#SBATCH --output=submit.out
#SBATCH --mem-per-cpu=32G
#SBATCH --mem-per-gpu=32G

module load OpenMPI/4.1.0-GCC-9.3.0
mpirun -q singularity exec --nv ../../../torch.simg python submit.py
