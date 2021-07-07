#!/bin/bash
#SBATCH -p pearl
#SBATCH --job-name=bvae
#SBATCH --time=5-00:00:00
#SBATCH -n 9
#SBATCH --gres=gpu:9
#SBATCH --output=submit.out

module load OpenMPI/4.1.0-GCC-9.3.0
mpirun -q singularity exec --nv ../../../torch.simg python submit.py
