import numpy as np
from pathlib import Path
import sys

if __name__ == '__main__':
    # constraint type
    cons = sys.argv[1]
    assert cons in ['kl', 'rec']

    # epochs
    epochs = 200

    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent.parent

    seed = 1234
    nlat = 10
    batchs = 64
    lrs = [0.0005, 0.001]
    increments = [4, 2]
    if cons == "rec":
        epsilon = 30
    else:
        epsilon = 0.08

    models = [f"evae_celeba/{cons}_{epochs}ep_z{nlat}_e{epsilon}_s{seed}_lr{lr}_incr{increment}"
              for lr in lrs
              for increment in increments]

    for model in models:
        print(f'{model}    ', end='')
        # check convergence
        loss_log = main_path / (f'results/{model}/test_losses.log')
        if not loss_log.exists():
            print('0', end='')
        else:
            print('1', end='')
        print('')
