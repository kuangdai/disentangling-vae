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
    # debug
    epochs_list = [1, 1, 1]
    # epochs_list = [120, 800, 1200]
    seed = 1234
    nlat = 64
    batchs = 64
    lr = 1e-5
    n_stddevs = 3

    datasets = ["dsprites", "celeba", "chairs"]
    alpha_gammas = [0.5, 1, 2]
    betas = [3, 6]

    # .sh filename
    fname = my_path / f'run_jobs_1.sh'

    # clear .sh file
    os.system(f'rm {fname}')

    # VAE
    for data, epochs in zip(datasets, epochs_list):

        VAE_cmd = (
<<<<<<< HEAD
            # f"python main.py qualitative/VAE_{data}_z{nlat} -s {seed} "
            # f"--checkpoint-every 50 -d {data} -e {epochs} -b {batchs} "
            # f"-z {nlat} -l VAE --lr {lr} "
            # f'--no-progress-bar -F {str(my_path / f"VAE_{data}_z{nlat}.out")} '
            # f"--record-loss-every=50 --pin-dataset-gpu \n"
=======
            f"python main.py qualitative/VAE_{data}_z{nlat} -s {seed} "
            f"--checkpoint-every 50 -d {data} -e {epochs} -b {batchs} "
            f"-z {nlat} -l VAE --lr {lr} "
            f'--no-progress-bar -F {str(my_path / f"out/VAE_{data}_z{nlat}.out")} '
            f"--record-loss-every=50 --pin-dataset-gpu \n"
>>>>>>> a23f20efcdb2b30a1d478cdc8684190e9dc5e691
            f"python main_viz.py qualitative/VAE_{data}_z{nlat} "
            f"all --is-show-loss --is-posterior -s {seed} --max-traversal {n_stddevs} \n"
        )

        alpha_gamma = 1
        beta = 1
        BTC_cmd = (
<<<<<<< HEAD
            # f"python main.py qualitative/btcvae_{data}_z{nlat}_A{alpha_gamma}_B{beta}_G{alpha_gamma} -s {seed} "
            # f"--checkpoint-every 50 -d {data} -e {epochs} -b {batchs} "
            # f"-z {nlat} -l btcvae --lr {lr} --btcvae-A {alpha_gamma} --btcvae-B {beta} --btcvae-G {alpha_gamma} "
            # f'--no-progress-bar -F {str(my_path / f"btcvae_{data}_z{nlat}_A{alpha_gamma}_B{beta}_G{alpha_gamma}.out")} '
            # f"--record-loss-every=50 --pin-dataset-gpu \n"
=======
            f"python main.py qualitative/btcvae_{data}_z{nlat}_A{alpha_gamma}_B{beta}_G{alpha_gamma} -s {seed} "
            f"--checkpoint-every 50 -d {data} -e {epochs} -b {batchs} "
            f"-z {nlat} -l btcvae --lr {lr} --btcvae-A {alpha_gamma} --btcvae-B {beta} --btcvae-G {alpha_gamma} "
            f'--no-progress-bar -F {str(my_path / f"out/btcvae_{data}_z{nlat}_A{alpha_gamma}_B{beta}_G{alpha_gamma}.out")} '
            f"--record-loss-every=50 --pin-dataset-gpu \n"
>>>>>>> a23f20efcdb2b30a1d478cdc8684190e9dc5e691
            f"python main_viz.py qualitative/btcvae_{data}_z{nlat}_A{alpha_gamma}_B{beta}_G{alpha_gamma} "
            f"all --is-show-loss --is-posterior -s {seed} --max-traversal {n_stddevs} \n"
        )

        with open(fname, 'a') as f:
            f.write(VAE_cmd)

    # beta-TCVAE
    for data, epochs in zip(datasets, epochs_list):
        for alpha_gamma in alpha_gammas:
            for beta in betas:

                BTC_cmd = (
<<<<<<< HEAD
                    # f"python main.py qualitative/btcvae_{data}_z{nlat}_A{alpha_gamma}_B{beta}_G{alpha_gamma} -s {seed} "
                    # f"--checkpoint-every 50 -d {data} -e {epochs} -b {batchs} "
                    # f"-z {nlat} -l btcvae --lr {lr} --btcvae-A {alpha_gamma} --btcvae-B {beta} --btcvae-G {alpha_gamma} "
                    # f'--no-progress-bar -F {str(my_path / f"btcvae_{data}_z{nlat}_A{alpha_gamma}_B{beta}_G{alpha_gamma}.out")} '
                    # f"--record-loss-every=50 --pin-dataset-gpu \n"
=======
                    f"python main.py qualitative/btcvae_{data}_z{nlat}_A{alpha_gamma}_B{beta}_G{alpha_gamma} -s {seed} "
                    f"--checkpoint-every 50 -d {data} -e {epochs} -b {batchs} "
                    f"-z {nlat} -l btcvae --lr {lr} --btcvae-A {alpha_gamma} --btcvae-B {beta} --btcvae-G {alpha_gamma} "
                    f'--no-progress-bar -F {str(my_path / f"out/btcvae_{data}_z{nlat}_A{alpha_gamma}_B{beta}_G{alpha_gamma}.out")} '
                    f"--record-loss-every=50 --pin-dataset-gpu \n"
>>>>>>> a23f20efcdb2b30a1d478cdc8684190e9dc5e691
                    f"python main_viz.py qualitative/btcvae_{data}_z{nlat}_A{alpha_gamma}_B{beta}_G{alpha_gamma} "
                    f"all --is-show-loss --is-posterior -s {seed} --max-traversal {n_stddevs} \n"
                )

                with open(fname, 'a') as f:
                    f.write(BTC_cmd)
