from pathlib import Path
import sys

import numpy as np

if __name__ == '__main__':
    # constraint type
    cons = sys.argv[1]
    assert cons in ['kl', 'rec']

    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent

    # read
    epses = np.loadtxt(my_path / f'grid_eps_{cons}')
    nlats = np.loadtxt(my_path / 'grid_nlats')
    seed = 0

    # jobs grouped by eps
    for ieps, eps in enumerate(epses):
        print(f'eps-{ieps:02d}    ', end='')
        for inlat, nlat in enumerate(nlats):
            log50 = main_path / (f'results/evae_dsprites_{cons}/z%d_e%s_s{seed}/' \
                                 f'train_losses_epoch49.log' % (nlat, str(eps)))
            print(int(log50.exists()), end='')
        print('')
