import numpy as np
from pathlib import Path
import sys, os

if __name__ == "__main__":
    """
    Jobs:
    1) M-EPS-VAE for data=[dsprites, celeba, chairs] with hypars derived from beta-TCVAE training loss
    """

    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent.parent

    # hypars
    epochs_list = [120, 800, 1200]
    seed = 1234
    nlat = 64
    batchs = 64
    lr = 1e-5
    n_stddevs = 3

    datasets = ["dsprites", "celeba", "chairs"]
    eps_As = [260, 265, 270, 275]
    eps_Gs = [1, 3, 5, 10]
    eps_Bs_dsprites_chairs = [-255, -260, -265, -270]
    eps_Bs_celeba = [-230, -235, -240, -245]

    # cherry-pick data samples as done in repo
    cherry_picked = ["92595 339150 656090", 
                     "40919 5172 22330", 
                     "88413 176606 179144 32260 191281 143307 101535 70059 87889 131612"]

    # .sh filename
    fname = my_path / f'run_jobs_2.sh'

    # clear .sh file
    os.system(f'rm {fname}')

    # multi-epsilon-VAE
    for data, epochs, cherries in zip(datasets, epochs_list, cherry_picked):
        if data=="celeba":
            eps_Bs = eps_Bs_celeba
        else:
            eps_Bs = eps_Bs_dsprites_chairs
        for eps_A in eps_As:
            for eps_G in eps_Gs:
                for eps_B in eps_Bs:

                    BTC_cmd = (
                        # f"python main.py qualitative/btcvae_{data}_z{nlat}_A{alpha_gamma}_B{beta}_G{alpha_gamma} -s {seed} "
                        # f"--checkpoint-every 50 -d {data} -e {epochs} -b {batchs} "
                        # f"-z {nlat} -l btcvae --lr {lr} --btcvae-A {alpha_gamma} --btcvae-B {beta} --btcvae-G {alpha_gamma} "
                        # f'--no-progress-bar -F {str(my_path / f"btcvae_{data}_z{nlat}_A{alpha_gamma}_B{beta}_G{alpha_gamma}.out")} '
                        # f"--record-loss-every=50 --pin-dataset-gpu \n"
                        f"python main_viz.py qualitative/btcvae_{data}_z{nlat}_A{alpha_gamma}_B{beta}_G{alpha_gamma} all -i {cherries} "
                        f"-s {seed} -c 10 -r 10 -t 2 --is-show-loss --is-posterior \n"
                    )

                    with open(fname, 'a') as f:
                        f.write(BTC_cmd)
