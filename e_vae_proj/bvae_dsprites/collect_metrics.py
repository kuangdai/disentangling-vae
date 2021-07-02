from pathlib import Path

import numpy as np
import torch


def axis_aligned_decay(sorted_mut_info, base=.9, num=10000):
    # normalize based on max
    sorted_mut_info /= sorted_mut_info[:, 0][:, None]
    # weights
    num = min(num, sorted_mut_info.shape[1])
    weights = base ** torch.arange(0, num)
    # sum with weights
    aad_k = (sorted_mut_info[:, 0:num] * weights).sum(dim=1) / weights.sum()
    aad_k[torch.isnan(aad_k)] = 0
    aad = aad_k.mean()
    return aad


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
    batchs_per_epoch = len(range(0, 737280, 256))

    # results
    LCM = np.zeros((len(betas), len(nlats)))
    MIG = np.zeros((len(betas), len(nlats)))
    AAM = np.zeros((len(betas), len(nlats)))
    AAD = np.zeros((len(betas), len(nlats)))
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

            # compute modified AAD
            metric_helpers = torch.load(res_dir / 'metric_helpers.pth')
            H_z = metric_helpers['marginal_entropies']
            H_zCv = metric_helpers['cond_entropies']
            mut_info = - H_zCv + H_z
            sorted_mut_info = torch.sort(mut_info, dim=1, descending=True)[
                0].clamp(min=0)
            AAD[ibeta, inlat] = axis_aligned_decay(sorted_mut_info)
            print(f'DONE: {res_dir}')

            # collect last epoch
            epoch = epochs - 1
            fname = res_dir / f'train_losses_epoch{epoch}.log'
            data = np.loadtxt(fname, skiprows=1)
            REC[ibeta, inlat] = data[:, 0].mean()
            KL[ibeta, inlat] = data[:, 1].mean()
            LOSS[ibeta, inlat] = data[:, -1].mean()
            print(f'DONE: {epoch}', end='\r')

    # save
    np.save(my_path / f'results/metric_LCM.npy', LCM)
    np.save(my_path / f'results/metric_MIG.npy', MIG)
    np.save(my_path / f'results/metric_AAM.npy', AAM)
    np.save(my_path / f'results/metric_AAD.npy', AAD)
    np.save(my_path / f'results/metric_REC.npy', REC)
    np.save(my_path / f'results/metric_KL.npy', KL)
    np.save(my_path / f'results/metric_LOSS.npy', LOSS)
