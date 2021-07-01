from pathlib import Path

import numpy as np

if __name__ == '__main__':
    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent

    # read
    epsilons = np.loadtxt(my_path / 'grid_epsilons')
    nlats = np.loadtxt(my_path / 'grid_nlats')
    seed = 0

    # jobs grouped by epsilon
    for iepsilon, epsilon in enumerate(epsilons):
        print(f'epsilon-{iepsilon:02d}    ', end='')
        for inlat, nlat in enumerate(nlats):
            log50 = main_path / (f'results/bvae_dsprites/z%d_b%s_s{seed}/' \
                                 f'train_losses_epoch49.log' % (nlat, str(epsilon)))
            print(int(log50.exists()), end='')
        print('')
