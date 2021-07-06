import numpy as np
import torch
from mpi4py import MPI

from ...main import parse_arguments, main


def main_device(argv, device=0, info_only=False):
    if device >= 0:
        torch.cuda.set_device(device)
    args = parse_arguments(argv)
    if info_only:
        name = torch.cuda.get_device_name(device)
        print(f"DEVICE {device} <{name}>;  JOB {argv}")
    else:
        main(args)


if __name__ == "__main__":
    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert comm.Get_size() == torch.cuda.device_count()

    # read
    betas = np.loadtxt('grid_betas')
    nlats = np.loadtxt('grid_nlats')
    seed = 0

    # argv template
    argv_tmp = f'bvae_dsprites/z%d_b%s_s{seed} -s {seed} ' \
               f'--checkpoint-every 10000 -d dsprites -e 50 -b 256 --lr 0.0003 ' \
               f'-z %d -l betaH --betaH-B %s --is-metrics --no-test ' \
               f'--no-progress-bar\n'

    # create bvae
    for ibeta, beta in enumerate(betas):
        if ibeta % rank == 0:
            for nlat in nlats:
                unnormalized_beta = beta * 64 * 64 / nlat
                argv = argv_tmp % (
                    nlat, str(beta), nlat, str(unnormalized_beta))
                main_device(argv, device=rank, info_only=True)
