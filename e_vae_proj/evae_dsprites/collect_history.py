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
    batchs_per_epoch = len(range(0, 737280, 256))

    # results
    hist_rec = np.zeros((len(epses), len(nlats), epochs, batchs_per_epoch))
    hist_KL = np.zeros((len(epses), len(nlats), epochs, batchs_per_epoch))
    hist_loss = np.zeros((len(epses), len(nlats), epochs, batchs_per_epoch))
    hist_lambda = np.zeros((len(epses), len(nlats), epochs, batchs_per_epoch))
    for ieps, eps in enumerate(epses):
        for inlat, nlat in enumerate(nlats):
            res_dir = main_path / (f'results/evae_dsprites/{cons}/'
                                   f'z%d_e%s_s{seed}/' % (nlat, str(eps)))
            # collect history
            for epoch in range(epochs):
                fname = res_dir / f'train_losses_epoch{epoch}.log'
                data = np.loadtxt(fname, skiprows=1)
                hist_rec[ieps, inlat, epoch, :] = data[:, 1]
                hist_KL[ieps, inlat, epoch, :] = data[:, 2]
                hist_loss[ieps, inlat, epoch, :] = data[:, -3]
                hist_lambda[ieps, inlat, epoch, :] = data[:, -2]
                print(f'DONE: {epoch}', end='\r')
            print(f'DONE: {res_dir}')

    # save
    np.save(my_path / f'results/{cons}/hist_rec.npy', hist_rec)
    np.save(my_path / f'results/{cons}/hist_KL.npy', hist_KL)
    np.save(my_path / f'results/{cons}/hist_loss.npy', hist_loss)
    np.save(my_path / f'results/{cons}/hist_lambda.npy', hist_lambda)
