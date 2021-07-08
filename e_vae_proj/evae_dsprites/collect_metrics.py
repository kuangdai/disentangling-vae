from pathlib import Path

import numpy as np
import torch
import sys


def axis_aligned_decay(sorted_mut_info):
    lat_sizes = np.array([3, 6, 40, 32, 32])
    H_v = torch.from_numpy(lat_sizes).float().log()

    n = sorted_mut_info.shape[1]
    KL_amp = np.log(n)
    first = sorted_mut_info[:, 0] / sorted_mut_info[:, :n].sum(dim=1)
    KL_first = - torch.log(first)
    aad_k = 1 - KL_first / KL_amp
    aad_k[torch.isnan(aad_k)] = 0

    aad_k *= sorted_mut_info[:, 0] / H_v
    return aad_k.mean()


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
    epochs = 49  # one blows up at 50
    batchs_per_epoch = len(range(0, 737280, 256))

    # results
    LCM = np.zeros((len(epses), len(nlats)))
    MIG = np.zeros((len(epses), len(nlats)))
    AAM = np.zeros((len(epses), len(nlats)))
    AAD = np.zeros((len(epses), len(nlats)))
    REC = np.zeros((len(epses), len(nlats)))
    KL = np.zeros((len(epses), len(nlats)))
    LOSS = np.zeros((len(epses), len(nlats)))
    LMBD = np.zeros((len(epses), len(nlats)))
    for ieps, eps in enumerate(epses):
        for inlat, nlat in enumerate(nlats):
            res_dir = main_path / (f'results/evae_dsprites_{cons}/'
                                   f'z%d_e%s_s{seed}/' % (nlat, str(eps)))
            try:
                # collect metrics
                with open(res_dir / 'metrics.log', 'r') as handle:
                    mdict = eval(handle.read().replace('NaN', '0.'))
                    LCM[ieps, inlat] = mdict['LCM']
                    MIG[ieps, inlat] = mdict['MIG']
                    AAM[ieps, inlat] = mdict['AAM']

                # compute modified AAD
                metric_helpers = torch.load(res_dir / 'metric_helpers.pth')
                H_z = metric_helpers['marginal_entropies']
                H_zCv = metric_helpers['cond_entropies']
                mut_info = - H_zCv + H_z
                sorted_mut_info = torch.sort(mut_info, dim=1, descending=True)[
                    0].clamp(min=0)
                AAD[ieps, inlat] = axis_aligned_decay(sorted_mut_info)

                # collect last epoch
                epoch = epochs - 1
                fname = res_dir / f'train_losses_epoch{epoch}.log'
                data = np.loadtxt(fname, skiprows=1)
                REC[ieps, inlat] = data[:, 1].mean()
                KL[ieps, inlat] = data[:, 2].mean()
                LOSS[ieps, inlat] = data[:, -3].mean()
                LMBD[ieps, inlat] = data[:, -2].mean()
                print(f'DONE: {res_dir}')
            except IOError:
                print(f'SKIP: {res_dir}')

    # save
    np.save(my_path / f'results_{cons}/metric_LCM.npy', LCM)
    np.save(my_path / f'results_{cons}/metric_MIG.npy', MIG)
    np.save(my_path / f'results_{cons}/metric_AAM.npy', AAM)
    np.save(my_path / f'results_{cons}/metric_AAD.npy', AAD)
    np.save(my_path / f'results_{cons}/metric_REC.npy', REC)
    np.save(my_path / f'results_{cons}/metric_KL.npy', KL)
    np.save(my_path / f'results_{cons}/metric_LOSS.npy', LOSS)
    np.save(my_path / f'results_{cons}/metric_LMBD.npy', LMBD)