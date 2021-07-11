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
    epochs = 50
    record_every = 50
    batchs = len(range(0, 737280 * epochs, 256))
    records = len(range(0, batchs, record_every))

    # results
    hist_rec = np.zeros((len(betas), len(nlats), records))
    hist_KL = np.zeros((len(betas), len(nlats), records))
    hist_loss = np.zeros((len(betas), len(nlats), records))
    for ibeta, beta in enumerate(betas):
        for inlat, nlat in enumerate(nlats):
            res_dir = main_path / (f'results/bvae_dsprites/'
                                   f'z%d_b%s_s{seed}/' % (nlat, str(beta)))
            # collect history
            row = 0
            for epoch in range(epochs):
                fname = res_dir / f'train_losses_epoch{epoch}.log'
                data = np.loadtxt(fname, skiprows=1)
                n = data.shape[0]
                hist_rec[ibeta, inlat, row:row + n] = data[:, 0]
                hist_KL[ibeta, inlat, row:row + n] = data[:, 1]
                hist_loss[ibeta, inlat, row:row + n] = data[:, -1]
                row += n
                print(f'DONE: {epoch}', end='\r')
            print(f'DONE: {res_dir}')

    # save
    np.save(my_path / f'results/hist_rec.npy', hist_rec)
    np.save(my_path / f'results/hist_KL.npy', hist_KL)
    np.save(my_path / f'results/hist_loss.npy', hist_loss)
