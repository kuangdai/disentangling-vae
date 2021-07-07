import numpy as np
from pathlib import Path
import sys

if __name__ == '__main__':

    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent

    seed = 0
    nlat = 10
    beta = 4.0
    epochs = 100

    # cmd template
    cmd = f'python main_viz.py bvaeH_mnist_{epochs}ep/z{nlat}_b{beta}_s{seed} ' \
              f'--is-show-loss --is-posterior' \
              f'--plots all ' \
              f'\n'

    with open(my_path / f'viz_beta{beta}_ep{epochs}.sh', 'w') as f:
        f.write(cmd)
