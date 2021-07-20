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


    # write to .sh file
    with open('scp_pretrained.sh', 'w') as f:
        for model in models:
            f.write(f'scp -r rsz68485@172.16.102.42:~/disentangling-vae/results/{model}/test_losses.log {main_path}/results/{model}/ \n')