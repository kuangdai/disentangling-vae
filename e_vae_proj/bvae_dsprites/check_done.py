import sys
from pathlib import Path

import numpy as np

if __name__ == '__main__':
    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent

    # read
    betas = np.loadtxt(my_path / 'grid_betas')
    nlats = np.loadtxt(my_path / 'grid_nlats')
    seed = 0
    epochs = 50

    # tolerance for convergence
    try:
        tol = float(sys.argv[1])
    except:
        tol = .5

    # jobs grouped by beta
    for ibeta, beta in enumerate(betas):
        print(f'beta-{ibeta:02d}    ', end='')
        for inlat, nlat in enumerate(nlats):
            metrics_log = main_path / (f'results/bvae_dsprites/z%d_b%s_s{seed}/'
                                       f'metrics.log' % (nlat, str(beta)))
            if not metrics_log.exists():
                print('0', end='')
            else:
                # check convergence
                loss_log = main_path / (
                        f'results/bvae_dsprites/z%d_b%s_s{seed}/'
                        f'train_losses_epoch{epochs - 1}.log' %
                        (nlat, str(beta)))
                loss = np.loadtxt(loss_log, skiprows=1)[:, -1]
                if (loss.max() - loss.mean()) / loss.mean() > tol:
                    print('_', end='')
                else:
                    print('1', end='')
        print('')
