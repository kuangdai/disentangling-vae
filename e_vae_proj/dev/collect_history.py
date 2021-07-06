from pathlib import Path

import numpy as np

if __name__ == '__main__':
    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent

    # read
    epsilons_kl = np.loadtxt(my_path.parent / 'evae_dsprites' / 'grid_eps_kl')
    epsilons_recon = np.loadtxt(my_path.parent / 'evae_dsprites' / 'grid_eps_rec')
    nlats = np.loadtxt(my_path.parent / 'evae_dsprites' / 'grid_nlats')
    seed = 0

    # sizes
    epochs = 10
    data_entries_per_epoch = 2880

    # arrays
    hist_rec = np.zeros((epochs, data_entries_per_epoch))
    hist_KL = np.zeros((epochs, data_entries_per_epoch))
    hist_loss = np.zeros((epochs, data_entries_per_epoch))
    hist_lambda = np.zeros((epochs, data_entries_per_epoch))

    # results
    res_dir = main_path / 'results/z10_e0.001_s0/'
    # collect history
    for epoch in range(epochs):
        fname = res_dir / f'train_losses_epoch{epoch}.log'
        data = np.loadtxt(fname, skiprows=1)
        hist_rec[epoch, :] = data[:, 1]
        hist_KL[epoch, :] = data[:, 2]
        hist_loss[epoch, :] = data[:, -3]
        hist_lambda[epoch, :] = data[:, -2]
        print(f'DONE: {epoch}', end='\r')
    print(f'DONE: {res_dir}')
    
    # save
    np.save(my_path / f'hist_constr_kl_z10_e0.001_s0_rec.npy', hist_rec)
    np.save(my_path / f'hist_constr_kl_z10_e0.001_s0_KL.npy', hist_KL)
    np.save(my_path / f'hist_constr_kl_z10_e0.001_s0_loss.npy', hist_loss)
    np.save(my_path / f'hist_constr_kl_z10_e0.001_s0_lambda.npy', hist_lambda)
    
    # results
    res_dir = main_path / 'results/z10_e3.5_s0/'
    # collect history
    for epoch in range(epochs):
        fname = res_dir / f'train_losses_epoch{epoch}.log'
        data = np.loadtxt(fname, skiprows=1)
        hist_rec[epoch, :] = data[:, 1]
        hist_KL[epoch, :] = data[:, 2]
        hist_loss[epoch, :] = data[:, -3]
        hist_lambda[epoch, :] = data[:, -2]
        print(f'DONE: {epoch}', end='\r')
    print(f'DONE: {res_dir}')

    # save
    np.save(my_path / f'hist_constr_kl_z10_e3.5_rec.npy', hist_rec)
    np.save(my_path / f'hist_constr_kl_z10_e3.5_KL.npy', hist_KL)
    np.save(my_path / f'hist_constr_kl_z10_e3.5_loss.npy', hist_loss)
    np.save(my_path / f'hist_constr_kl_z10_e3.5_lambda.npy', hist_lambda)
