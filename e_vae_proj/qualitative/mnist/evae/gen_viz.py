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
    cmd_tmp = f'python main_viz.py evae_mnist_{cons}_{epochs}ep/z%d_e%s_s{seed} ' \
              f'--is-show-loss --is-posterior' \
              f'--plots all ' \
              f'\n'

    with open(my_path / f'viz_{cons}_eps{epsilon}.sh', 'w') as f:
        if cons == 'rec':
            unnormalized_eps = epsilon * 64 * 64
        else:
            unnormalized_eps = epsilon * nlat
        cmd = cmd_tmp % (nlat, str(epsilon), nlat, str(unnormalized_eps))
        f.write(cmd)
