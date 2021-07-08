import numpy as np
from pathlib import Path
import sys

if __name__ == '__main__':

    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent

    seed = 0
    nlat = 10
    alpha = 1.0
    beta = 6.0
    gamma = 1.0
    epochs = 100

    # cmd template
    cmd = f'python main.py btcvae_mnist_{epochs}ep/z{nlat}_a{alpha}_b{beta}_g{gamma}_s{seed} -s {seed} ' \
              f'--checkpoint-every 25 -d mnist -e {epochs} -b 64 --lr 0.0005 ' \
              f'-z {nlat} -l btcvae --btcvae-A {alpha} --btcvae-B {beta} --btcvae-G {gamma} ' \
              f'--no-test\n'

    with open(my_path / f'train_beta{beta}.sh', 'w') as f:
        unnormalized_beta = beta * nlat
        f.write(cmd)
