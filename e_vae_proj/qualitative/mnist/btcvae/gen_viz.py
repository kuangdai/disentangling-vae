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
    cmd = f'python main_viz.py btcvae_mnist_{epochs}ep/z{nlat}_a{alpha}_b{beta}_g{gamma}_s{seed} ' \
              f'all ' \
              f'--is-show-loss --is-posterior ' \
              f'\n'

    with open(my_path / f'viz_beta{beta}_ep{epochs}.sh', 'w') as f:
        f.write(cmd)
