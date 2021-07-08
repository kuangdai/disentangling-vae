import os
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI


def run(rank, size, comm):
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
               f'--no-progress-bar -F {str(my_path / f"results/z%d_b%s_s{seed}.out")} ' \
               f'--gpu-id={rank}'

    # change dir
    os.chdir(main_path)
    os.system('rm -rf results/bvae_dsprites/*')
    comm.Barrier()

    # create bvae
    for ibeta, beta in enumerate(betas):
        if ibeta % size == rank:
            for nlat in nlats:
                unnormalized_beta = beta * 64 * 64 / nlat
                argv = argv_tmp % (
                    nlat, str(beta), nlat, str(unnormalized_beta),
                    nlat, str(beta))
                args = parse_arguments(argv.split(' '))
                print(f'RANK {rank}, SIZE {size}, JOB {argv.split(" ")[0]}')
                main(args)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()
    run(world_rank, world_size, comm)
