from pathlib import Path

import numpy as np
import torch


if __name__ == '__main__':
    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent

    # read
    betas = np.loadtxt(my_path / 'grid_betas')
    nlats = np.loadtxt(my_path / 'grid_nlats')
    seed = 0

    # results
    LCM = np.zeros((len(betas), len(nlats)))
    MIG = np.zeros((len(betas), len(nlats)))
    AAM = np.zeros((len(betas), len(nlats)))
    MID = np.zeros((len(betas), len(nlats)))
    REC = np.zeros((len(betas), len(nlats)))
    KL = np.zeros((len(betas), len(nlats)))
    LOSS = np.zeros((len(betas), len(nlats)))
    for ibeta, beta in enumerate(betas):
        for inlat, nlat in enumerate(nlats):
            res_dir = main_path / (f'results/bvae_dsprites/'
                                   f'z%d_b%s_s{seed}/' % (nlat, str(beta)))
            # collect metrics
            with open(res_dir / 'metrics.log', 'r') as handle:
                mdict = eval(handle.read())
                LCM[ibeta, inlat] = mdict['LCM']
                MIG[ibeta, inlat] = mdict['MIG']
                AAM[ibeta, inlat] = mdict['AAM']
                MID[ibeta, inlat] = mdict['MID']

            # collect last epoch
            epochs = 50
            fname = res_dir / f'train_losses_epoch{epochs - 1}.log'
            data = np.loadtxt(fname, skiprows=1)
            REC[ibeta, inlat] = data[:, 0].mean()
            KL[ibeta, inlat] = data[:, 1].mean()
            LOSS[ibeta, inlat] = data[:, -1].mean()
            print(f'DONE: {res_dir}')

    # save
    np.save(my_path / f'results/metric_LCM.npy', LCM)
    np.save(my_path / f'results/metric_MIG.npy', MIG)
    np.save(my_path / f'results/metric_AAM.npy', AAM)
    np.save(my_path / f'results/metric_MID.npy', MID)
    np.save(my_path / f'results/metric_REC.npy', REC)
    np.save(my_path / f'results/metric_KL.npy', KL)
    np.save(my_path / f'results/metric_LOSS.npy', LOSS)
