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
    eps_A = 270
    eps_G = 5
    eps_Bs_dsprites_chairs = [-255, -260, -265, -270]
    eps_Bs_celeba = [-230, -235, -240, -245]

    # cherry-pick data samples as done in repo
    cherry_picked = ["92595 339150 656090", 
                     "88413 176606 179144 32260 191281 143307 101535 70059 87889 131612",
                     "40919 5172 22330", ]

    # .sh filename
    fname = my_path / f'run_jobs_2.sh'

    # clear .sh file
    os.system(f'rm {fname}')


    # multi-epsilon-VAE from chaos
    for data, epochs, cherries in zip(datasets, epochs_list, cherry_picked):
        if data=="celeba":
            eps_Bs = eps_Bs_celeba
        else:
            eps_Bs = eps_Bs_dsprites_chairs
        for eps_B in eps_Bs:

            cmd = (
                f"python main.py qualitative/mepsvae_{data}_z{nlat}_epsA{eps_A}_epsB{eps_B}_epsG{eps_G} -s {seed} "
                f"--checkpoint-every 50 -d {data} -e {epochs} -b {batchs} "
                f"-z {nlat} -l mepsvae --lr {lr} --mepsvae-epsilon-alpha {eps_A} --mepsvae-epsilon-beta {eps_B} --mepsvae-epsilon-gamma {eps_G} "
                f'--no-progress-bar -F {str(my_path / f"mepsvae_{data}_z{nlat}_epsA{eps_A}_epsB{eps_B}_epsG{eps_G}.out")} '
                f"--record-loss-every=50 --pin-dataset-gpu \n"
                f"python main_viz.py qualitative/mepsvae_{data}_z{nlat}_epsA{eps_A}_epsB{eps_B}_epsG{eps_G} all -i {cherries} "
                f"-s {seed} -c 10 -r 10 -t 2 --is-show-loss --is-posterior \n"
            )

            with open(fname, 'a') as f:
                f.write(cmd)


    # multi-epsilon-VAE from VAE
    for data, epochs, cherries in zip(datasets, epochs_list, cherry_picked):
        if data=="celeba":
            eps_Bs = eps_Bs_celeba
        else:
            eps_Bs = eps_Bs_dsprites_chairs
        for eps_B in eps_Bs:

            cmd = (
                f"python main.py qualitative/mepsvae_fromVAE_{data}_z{nlat}_epsA{eps_A}_epsB{eps_B}_epsG{eps_G} -s {seed} "
                f"--checkpoint-every 50 -d {data} -e {epochs} -b {batchs} --mepsvae-warmup 0 --continue-train-from VAE_{data}_z64 "
                f"-z {nlat} -l mepsvae --lr {lr} --mepsvae-epsilon-alpha {eps_A} --mepsvae-epsilon-beta {eps_B} --mepsvae-epsilon-gamma {eps_G} "
                f'--no-progress-bar -F {str(my_path / f"mepsvae_fromVAE_{data}_z{nlat}_epsA{eps_A}_epsB{eps_B}_epsG{eps_G}.out")} '
                f"--record-loss-every=50 --pin-dataset-gpu \n"
                f"python main_viz.py qualitative/mepsvae_fromVAE_{data}_z{nlat}_epsA{eps_A}_epsB{eps_B}_epsG{eps_G} all -i {cherries} "
                f"-s {seed} -c 10 -r 10 -t 2 --is-show-loss --is-posterior \n"
            )

            with open(fname, 'a') as f:
                f.write(cmd)
