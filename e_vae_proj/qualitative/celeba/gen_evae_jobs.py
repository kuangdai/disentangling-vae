import numpy as np
from pathlib import Path
import sys

if __name__ == '__main__':
    # constraint type
    cons = sys.argv[1]
    assert cons in ['kl', 'rec']

    # epochs
    epochs = int(sys.argv[2])

    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent

    seed = 1234
    nlat = 10
    batchs = 64
    lrs = [0.0005, 0.001]
    increments = [4, 2]
    if cons == "rec":
        epsilon = 30
    else:
        epsilon = 0.08

    for lr in lrs:
        for increment in increments:
            # training cmd template
            cmd_train_tmp = f'python main.py evae_celeba/{cons}_{epochs}ep_z{nlat}_e{epsilon}_s{seed}_lr{lr}_incr{increment} -s {seed} ' \
                    f'--checkpoint-every 25 -d celeba -e {epochs} -b {batchs} --lr {lr} ' \
                    f'-z {nlat} -l epsvae %s ' \
                    f'--epsvae-epsilon %s --epsvae-interval-incr-L {increment} --no-test --record-loss-every=50 --pin-dataset-gpu \n'

            # eval cmd template
            cmd_eval = f'python main.py evae_celeba/{cons}_{epochs}ep_z{nlat}_e{epsilon}_s{seed} ' \
                    f'--is-eval-only \n'

            # viz cmd template
            cmd_viz = f'python main_viz.py evae_celeba/{cons}_{epochs}ep_z{nlat}_e{epsilon}_s{seed} ' \
                    f'all ' \
                    f'--is-show-loss --is-posterior -s {seed} ' \
                    f'\n' \
                    f'python main_viz.py evae_celeba/{cons}_{epochs}ep_z{nlat}_e{epsilon}_s{seed} ' \
                    f'traversals ' \
                    f'--is-show-loss -s {seed} ' \
                    f'\n'

            with open(my_path / f'train_eval_viz_{cons}_{epochs}ep_nlat{nlat}_eps{epsilon}_s{seed}_b{batchs}.sh', 'a') as f:
                if cons == 'rec':
                    unnormalized_eps = epsilon
                    cons_arg = "--epsvae-constrain-reconstruction"
                else:
                    unnormalized_eps = epsilon
                    cons_arg = "--epsvae-constrain-kl-divergence"
                cmd_train = cmd_train_tmp % (cons_arg, str(unnormalized_eps))
                f.write(cmd_train + cmd_eval + cmd_viz)
