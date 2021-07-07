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
    cmd = f'python main.py bvaeH_mnist_{epochs}ep/z{nlat}_b{beta}_s{seed} -s {seed} ' \
              f'--checkpoint-every 25 -d mnist -e {epochs} -b 64 --lr 0.0005 ' \
              f'-z {nlat} -l betaH --betaH-B {beta} ' \
              f'--is-metrics --no-test\n'

    with open(my_path / f'train_beta{beta}_ep{epochs}.sh', 'w') as f:
        f.write(cmd)
