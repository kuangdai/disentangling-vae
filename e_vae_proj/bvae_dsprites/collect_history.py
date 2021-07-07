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

    # sizes
    epochs = 45
    batchs_per_epoch = len(range(0, 737280, 256))

    # results
    hist_rec = np.zeros((len(betas), len(nlats), epochs, batchs_per_epoch))
    hist_KL = np.zeros((len(betas), len(nlats), epochs, batchs_per_epoch))
    hist_loss = np.zeros((len(betas), len(nlats), epochs, batchs_per_epoch))
    for ibeta, beta in enumerate(betas):
        for inlat, nlat in enumerate(nlats):
            res_dir = main_path / (f'results/bvae_dsprites/'
                                   f'z%d_b%s_s{seed}/' % (nlat, str(beta)))
            # collect history
            for epoch in range(epochs):
                fname = res_dir / f'train_losses_epoch{epoch}.log'
                data = np.loadtxt(fname, skiprows=1)
                hist_rec[ibeta, inlat, epoch, :] = data[:, 0]
                hist_KL[ibeta, inlat, epoch, :] = data[:, 1]
                hist_loss[ibeta, inlat, epoch, :] = data[:, -1]
                print(f'DONE: {epoch}', end='\r')
            print(f'DONE: {res_dir}')

    # save
    np.save(my_path / f'results/hist_rec.npy', hist_rec)
    np.save(my_path / f'results/hist_KL.npy', hist_KL)
    np.save(my_path / f'results/hist_loss.npy', hist_loss)
