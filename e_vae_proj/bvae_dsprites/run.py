import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def run(rank, size):
    # path and main
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent
    sys.path.insert(1, str(main_path))
    from main import parse_arguments, main

    # read
    betas = np.loadtxt(my_path / 'grid_betas')
    nlats = np.loadtxt(my_path / 'grid_nlats')
    seed = 0
    epochs = 50

    # argv template
    argv_tmp = f'bvae_dsprites/z%d_b%s_s{seed} -s {seed} ' \
               f'--checkpoint-every 10000 -d dsprites -e {epochs} -b 256 --lr 0.0003 ' \
               f'-z %d -l betaH --betaH-B %s --is-metrics --no-test ' \
               f'--no-progress-bar -F {str(my_path / "results/z%d_b%s_s{seed}.out")}'

    # change dir
    os.chdir(main_path)

    # create bvae
    for ibeta, beta in enumerate(betas):
        if ibeta % size == rank:
            for nlat in nlats:
                unnormalized_beta = beta * 64 * 64 / nlat
                argv = argv_tmp % (
                    nlat, str(beta), nlat, str(unnormalized_beta), nlat, str(beta))
                args = parse_arguments(argv.split(' '))
                main(args)


def init_processes(rank, size, gpu, fn, backend='mpi'):
    """ Initialize the distributed environment. """
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    init_processes(world_rank, world_size, gpu, run, backend='mpi')
