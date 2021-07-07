import os
import sys
from pathlib import Path

import numpy as np
import torch
from mpi4py import MPI

# ENV
my_path = Path(__file__).parent.resolve().expanduser()
main_path = my_path.parent.parent
sys.path.insert(1, str(main_path))
from main import parse_arguments, main


def main_device(argv, device=0, cuda=True):
    if cuda:
        torch.cuda.set_device(device)
    args = parse_arguments(argv.split(' '))
    name = torch.cuda.get_device_name(device) if cuda else 'cpu'
    print(f"DEVICE {device} <{name}>: {argv}")
    main(args)


if __name__ == "__main__":
    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    assert size == torch.cuda.device_count()

    # read
    betas = np.loadtxt(my_path / 'grid_betas')
    nlats = np.loadtxt(my_path / 'grid_nlats')
    seed = 0
    epochs = 45

    # argv template
    argv_tmp = f'bvae_dsprites/z%d_b%s_s{seed} -s {seed} ' \
               f'--checkpoint-every 10000 -d dsprites -e {epochs} -b 256 --lr 0.0003 ' \
               f'-z %d -l betaH --betaH-B %s --is-metrics --no-test ' \
               f'--no-progress-bar'

    # change dir
    os.chdir(main_path)

    # create bvae
    for ibeta, beta in enumerate(betas):
        if ibeta % size == rank:
            for nlat in nlats:
                unnormalized_beta = beta * 64 * 64 / nlat
                argv = argv_tmp % (
                    nlat, str(beta), nlat, str(unnormalized_beta))
                main_device(argv, device=rank)
