import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from pathlib import Path

def read_loss_from_file(log_file_path, loss_to_fetch):
    """ Read the average KL per latent dimension at the final stage of training from the log file.
        Parameters
        ----------
        log_file_path : str
            Full path and file name for the log file. For example 'experiments/custom/losses.log'.
        loss_to_fetch : str
            The loss type to search for in the log file and return. This must be in the exact form as stored.
    """
    EPOCH = "Epoch"
    LOSS = "Loss"

    logs = pd.read_csv(log_file_path)
    df_last_epoch_loss = logs[logs.loc[:, EPOCH] == logs.loc[:, EPOCH].max()]
    df_last_epoch_loss = df_last_epoch_loss.loc[df_last_epoch_loss.loc[:, LOSS].str.startswith(loss_to_fetch), :]
    df_last_epoch_loss.loc[:, LOSS] = df_last_epoch_loss.loc[:, LOSS].str.replace(loss_to_fetch, "").astype(int)
    df_last_epoch_loss = df_last_epoch_loss.sort_values(LOSS).loc[:, "Value"]
    return list(df_last_epoch_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", required=True, type=str, help="parent folder of the training results")
    args = parser.parse_args()

    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent

    # get loss data
    recon_loss = read_loss_from_file(Path(main_path) / 'results' / args.folder / 'train_losses.log', 'recon_loss')
    print("recon: \n", recon_loss)

    kl_loss = read_loss_from_file(Path(main_path) / 'results' / args.folder  / 'train_losses.log', 'kl_loss')
    print("kl: \n", kl_loss)

    lambda_vals = read_loss_from_file(Path(main_path) / 'results' / args.folder  / 'train_losses.log', 'lambda')
    print("lambda: \n", lambda_vals)