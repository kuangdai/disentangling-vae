import os
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI


def run(rank, size, comm, cons):
    # path and main
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent
    sys.path.insert(1, str(main_path))
    from main import parse_arguments, main

    # read
    epses = np.loadtxt(my_path / f'grid_eps_{cons}')
    nlats = np.loadtxt(my_path / 'grid_nlats')
    seed = 0
    epochs = 50

    # argv template
    argv_tmp = f'evae_dsprites/{cons}/z%d_e%s_s{seed} -s {seed} ' \
               f'--checkpoint-every 10000 -d dsprites -e {epochs} -b 256 --lr 0.0003 ' \
               f'-z %d -l epsvae --epsvae-constrain-%s --epsvae-epsilon %s --epsvae-lambda-lr 0.0003 ' \
               f'--is-metrics --no-test ' \
               f'--no-progress-bar -F {str(my_path / f"results/{cons}/z%d_e%s_s{seed}.out")} ' \
               f'--gpu-id={rank} --record-loss-every=50 --pin-dataset-gpu'

    # change dir
    os.chdir(main_path)
    os.system(f'rm -rf results/evae_dsprites/{cons}/*')
    comm.Barrier()

    # create evae
    for ieps, eps in enumerate(epses):
        if ieps % size == rank:
            for nlat in nlats:
                if cons == 'rec':
                    unnormalized_eps = eps * 64 * 64
                    cons_full = 'reconstruction'
                else:
                    unnormalized_eps = eps * nlat
                    cons_full = 'kl-divergence'
                argv = argv_tmp % (
                    nlat, str(eps), nlat, cons_full, str(unnormalized_eps),
                    nlat, str(eps))
                args = parse_arguments(argv.split(' '))
                print(f'RANK {rank}, SIZE {size}, JOB {argv.split(" ")[0]}')
                try:
                    main(args)
                except:
                    print(f'ERROR: RANK {rank}, SIZE {size}, JOB {argv.split(" ")[0]}')


if __name__ == "__main__":
    # constraint type
    cons = sys.argv[1]
    assert cons in ['kl', 'rec']

    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()
    run(world_rank, world_size, comm, cons)
