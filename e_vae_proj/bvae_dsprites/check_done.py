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

    # jobs grouped by beta
    for ibeta, beta in enumerate(betas):
        print(f'beta-{ibeta:02d}    ', end='')
        for inlat, nlat in enumerate(nlats):
            metrics_log = main_path / (f'results/bvae_dsprites/z%d_b%s_s{seed}/' \
                                       f'metrics.log' % (nlat, str(beta)))
            print(int(metrics_log.exists()), end='')
        print('')
