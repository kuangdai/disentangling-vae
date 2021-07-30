import sys
from pathlib import Path


if __name__ == '__main__':
    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent

    datasets = ["dsprites", "celeba", "chairs"]
    alpha_gammas = [0.5, 1, 2]
    betas = [3, 6]
    epochs_list = [120, 800, 1200]

    models = []
    for data, epochs in zip(datasets, epochs_list):
        models.append('qualitative/VAE_{data}_z{nlat}')
    for data, epochs in zip(datasets, epochs_list):
        alpha_gamma = 1
        beta = 1
        models.append('qualitative/btcvae_{data}_z{nlat}_A{alpha_gamma}_B{beta}_G{alpha_gamma}')
    
    for data, epochs in zip(datasets, epochs_list):
        for alpha_gamma in alpha_gammas:
            for beta in betas:
                models.append('qualitative/btcvae_{data}_z{nlat}_A{alpha_gamma}_B{beta}_G{alpha_gamma}')

    for model in models:
        print(f'{model}        ', end='')
        # check convergence
        loss_log = main_path / (f'results/{model}/train_losses_epoch{epochs-1}.log')
        if not loss_log.exists():
            print('0', end='')
        else:
            print('1', end='')
        print('')