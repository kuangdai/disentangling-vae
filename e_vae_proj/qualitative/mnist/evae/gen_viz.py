import numpy as np
from pathlib import Path
import sys

if __name__ == '__main__':
    # constraint type
    cons = sys.argv[1]
    assert cons in ['kl', 'rec']

    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent

    seed = 0
    nlat = 10
    epsilon = 1.0
    epochs = 100

    # cmd template
    cmd = f'python main_viz.py evae_mnist/{cons}_{epochs}ep_z{nlat}_e{epsilon}_s{seed} ' \
              f'all ' \
              f'--is-show-loss --is-posterior ' \
              f'\n'

    with open(my_path / f'viz_{cons}_eps{epsilon}.sh', 'w') as f:
        f.write(cmd)
