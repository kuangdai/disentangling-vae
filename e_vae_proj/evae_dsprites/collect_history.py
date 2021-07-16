import sys
from pathlib import Path

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

    # sizes
    epochs = 50
    record_every = 50
    batchs = len(range(0, 737280 * epochs, 256))
    records = len(range(0, batchs, record_every))

    # results
    hist_rec = np.zeros((len(epses), len(nlats), epochs, records))
    hist_KL = np.zeros((len(epses), len(nlats), epochs, records))
    hist_loss = np.zeros((len(epses), len(nlats), epochs, records))
    hist_lambda = np.zeros((len(epses), len(nlats), epochs, records))
    for ieps, eps in enumerate(epses):
        for inlat, nlat in enumerate(nlats):
            res_dir = main_path / (f'results/evae_dsprites/{cons}/'
                                   f'z%d_e%s_s{seed}/' % (nlat, str(eps)))
            # collect history
            row = 0
            for epoch in range(epochs):
                fname = res_dir / f'train_losses_epoch{epoch}.log'
                data = np.loadtxt(fname, skiprows=1)
                n = data.shape[0]
                hist_rec[ieps, inlat, row:row + n] = data[:, 1]
                hist_KL[ieps, inlat, row:row + n] = data[:, 2]
                hist_loss[ieps, inlat, row:row + n] = data[:, -3]
                hist_lambda[ieps, inlat, row:row + n] = data[:, -2]
                row += n
                print(f'DONE: {epoch}', end='\r')
            print(f'DONE: {res_dir}')

    # save
    np.save(my_path / f'results/{cons}/hist_rec.npy', hist_rec)
    np.save(my_path / f'results/{cons}/hist_KL.npy', hist_KL)
    np.save(my_path / f'results/{cons}/hist_loss.npy', hist_loss)
    np.save(my_path / f'results/{cons}/hist_lambda.npy', hist_lambda)
