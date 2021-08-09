import numpy as np
from pathlib import Path
import sys, os

if __name__ == "__main__":
    """
    Jobs:
    1) VAE (VAE loss) for data=[dsprites, celeba, chairs]
    2) VAE (beta-TC loss with alpha=beta=gamma=1) for data=[dsprites, celeba, chairs]
    3) beta-TCVAE for alpha=gamma=[0.5, 1, 2], for beta=[3,6], for data=[dsprites, celeba, chairs]
    """

    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent.parent

    # hypars
    cons_list = ["kl", "rec"]
    epochs_list = [120, 800, 1200]
    seed = 1234
    nlat = 64
    batchs = 64
    lr = 1e-5
    n_stddevs = 3

    datasets = ["dsprites", "celeba", "chairs"]
    alpha_gammas = [0.5, 1, 2]
    betas = [3, 6]

    # cherry-pick data samples as done in repo
    cherry_picked = ["92595 339150 656090", 
                     "88413 176606 179144 32260 191281 143307 101535 70059 87889 131612",
                     "40919 5172 22330", ]

    # .sh filename
    fname = my_path / f'run_jobs_1.sh'

    # clear .sh file
    os.system(f'rm {fname}')

    # VAE
    for data, epochs, cherries in zip(datasets, epochs_list, cherry_picked):

        VAE_cmd = (
            # f"python main.py qualitative/VAE_{data}_z{nlat} -s {seed} "
            # f"--checkpoint-every 50 -d {data} -e {epochs} -b {batchs} "
            # f"-z {nlat} -l VAE --lr {lr} "
            # f'--no-progress-bar -F {str(my_path / f"VAE_{data}_z{nlat}.out")} '
            # f"--record-loss-every=50 --pin-dataset-gpu \n"
            f"python main_viz.py qualitative/VAE_{data}_z{nlat} all -i {cherries} "
            f"-s {seed} -c 10 -r 10 -t 2 --is-show-loss --is-posterior \n"
        )

        alpha_gamma = 1
        beta = 1
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
            f.write(VAE_cmd + BTC_cmd)

    # beta-TCVAE
    for data, epochs, cherries in zip(datasets, epochs_list, cherry_picked):
        for alpha_gamma in alpha_gammas:
            for beta in betas:

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
