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

    # results
    LCM = np.zeros((len(epses), len(nlats)))
    MIG = np.zeros((len(epses), len(nlats)))
    AAM = np.zeros((len(epses), len(nlats)))
    MID = np.zeros((len(epses), len(nlats)))
    REC = np.zeros((len(epses), len(nlats)))
    KL = np.zeros((len(epses), len(nlats)))
    LOSS = np.zeros((len(epses), len(nlats)))
    LMBD = np.zeros((len(epses), len(nlats)))
    for ieps, eps in enumerate(epses):
        for inlat, nlat in enumerate(nlats):
            res_dir = main_path / (f'results/evae_dsprites/{cons}/'
                                   f'z%d_e%s_s{seed}/' % (nlat, str(eps)))
            # collect metrics
            with open(res_dir / 'metrics.log', 'r') as handle:
                mdict = eval(handle.read())
                LCM[ieps, inlat] = mdict['LCM']
                MIG[ieps, inlat] = mdict['MIG']
                AAM[ieps, inlat] = mdict['AAM']
                MID[ieps, inlat] = mdict['MID']

            # collect last epoch
            epochs = 50
            fname = res_dir / f'train_losses_epoch{epochs - 1}.log'
            data = np.loadtxt(fname, skiprows=1)
            REC[ieps, inlat] = data[:, 1].mean()
            KL[ieps, inlat] = data[:, 2].mean()
            LOSS[ieps, inlat] = data[:, -3].mean()
            LMBD[ieps, inlat] = data[:, -2].mean()
            print(f'DONE: {res_dir}')

    # save
    np.save(my_path / f'results/{cons}/metric_LCM.npy', LCM)
    np.save(my_path / f'results/{cons}/metric_MIG.npy', MIG)
    np.save(my_path / f'results/{cons}/metric_AAM.npy', AAM)
    np.save(my_path / f'results/{cons}/metric_MID.npy', MID)
    np.save(my_path / f'results/{cons}/metric_REC.npy', REC)
    np.save(my_path / f'results/{cons}/metric_KL.npy', KL)
    np.save(my_path / f'results/{cons}/metric_LOSS.npy', LOSS)
    np.save(my_path / f'results/{cons}/metric_LMBD.npy', LMBD)
