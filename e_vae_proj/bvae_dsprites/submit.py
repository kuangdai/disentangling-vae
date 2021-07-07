import os
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(size))
import torch
assert size == torch.cuda.device_count()
torch.cuda.set_device(rank)
torch.set_num_threads(1)

# path and main
my_path = Path(__file__).parent.resolve().expanduser()
main_path = my_path.parent.parent
sys.path.insert(1, str(main_path))
from main import parse_arguments, main


if __name__ == "__main__":
    # read
    betas = np.loadtxt(my_path / 'grid_betas')
    nlats = np.loadtxt(my_path / 'grid_nlats')
    seed = 0
    epochs = 50

    # argv template
    argv_tmp = f'bvae_dsprites/z%d_b%s_s{seed} -s {seed} ' \
               f'--checkpoint-every 10000 -d dsprites -e {epochs} -b 256 --lr 0.0003 ' \
               f'-z %d -l betaH --betaH-B %s --is-metrics --no-test ' \
               f'--no-progress-bar'

    # change dir
    os.chdir(main_path)

    # barrier
    comm.Barrier()

    # create bvae
    for ibeta, beta in enumerate(betas):
        if ibeta % size == rank:
            for nlat in nlats:
                unnormalized_beta = beta * 64 * 64 / nlat
                argv = argv_tmp % (
                    nlat, str(beta), nlat, str(unnormalized_beta))
                args = parse_arguments(argv.split(' '))
                main(args)
