import sys
from pathlib import Path


if __name__ == '__main__':
    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent

    LOSSES = ['VAE', 'betaH', 'betaB', 'factor', 'btcvae']
    DATASETS = ['mnist', 'celeba', 'chairs', 'dsprites']

    models = ["{}_{}".format(loss, data)
              for loss in LOSSES
              for data in DATASETS]

    for model in models:
        print(f'{model}    ', end='')
        # check convergence
        loss_log = main_path / (f'results/{model}/test_losses.log')
        if not loss_log.exists():
            print('0', end='')
        else:
            print('1', end='')
        print('')
