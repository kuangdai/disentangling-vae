import imageio
import logging
import os
from timeit import default_timer
from collections import defaultdict

import numpy as np

from tqdm import trange
import torch
from torch.nn import functional as F

from disvae.utils.modelIO import save_model
from torch.utils.data import DataLoader


TRAIN_LOSSES_LOGFILE = "train_losses.log"


class Trainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    optimizer: torch.optim.Optimizer

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    gif_visualizer : viz.Visualizer, optional
        Gif Visualizer that should return samples at every epochs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, optimizer, loss_f,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 gif_visualizer=None,
                 is_progress_bar=True):

        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.losses_logger = LossesLogger(os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE))
        self.gif_visualizer = gif_visualizer
        self.logger.info("Training Device: {}".format(self.device))

    def __call__(self, data_loader,
                 epochs=10,
                 checkpoint_every=10,
                 pin_dataset_gpu=False):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int, optional
            Number of epochs to train the model for.

        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.

        pin_dataset_gpu: bool, optional
            Pin entire dataset on GPU.
        """
        start = default_timer()
        self.model.train()

        N = len(data_loader.dataset)
        if pin_dataset_gpu:
            new_loader = DataLoader(data_loader.dataset,
                                    batch_size=N, shuffle=False)
            pinned_data, _ = next(iter(new_loader))
            # send all to GPU
            pinned_data = pinned_data.to(self.device)
        else:
            pinned_data = None

        for epoch in range(epochs):
            storer = defaultdict(list)
            if pinned_data is not None:
                pinned_data = pinned_data[torch.randperm(N)]  # shuffle
            mean_epoch_loss = self._train_epoch(data_loader, storer, epoch, pinned_data)
            self.logger.info('Epoch: {} Average loss per image: {:.2f}'.format(epoch + 1,
                                                                               mean_epoch_loss))
            self.losses_logger.log(epoch, storer)

            if self.gif_visualizer is not None:
                self.gif_visualizer()

            if epoch % checkpoint_every == 0:
                save_model(self.model, self.save_dir,
                           filename="model-{}.pt".format(epoch))

        if pinned_data is not None:
            del pinned_data
            torch.cuda.empty_cache()

        if self.gif_visualizer is not None:
            self.gif_visualizer.save_reset()

        self.model.eval()

        delta_time = (default_timer() - start) / 60
        self.logger.info('Finished training after {:.1f} min.'.format(delta_time))

    def _train_epoch(self, data_loader, storer, epoch, pinned_data=None):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        storer: dict
            Dictionary in which to store important variables for vizualisation.

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
            Mean loss per image
        """
        epoch_loss = 0.
        kwargs = dict(desc="Epoch {}".format(epoch + 1), leave=False,
                      disable=not self.is_progress_bar)
        if pinned_data is not None:
            N = len(data_loader.dataset)
            with trange(len(data_loader), **kwargs) as t:
                for start in range(0, N, data_loader.batch_size):
                    end = min(N, start + data_loader.batch_size)
                    iter_loss = self._train_iteration(pinned_data[start:end], storer)
                    epoch_loss += iter_loss

                    t.set_postfix(loss=iter_loss)
                    t.update()
        else:
            with trange(len(data_loader), **kwargs) as t:
                for _, (data, _) in enumerate(data_loader):
                    iter_loss = self._train_iteration(data, storer)
                    epoch_loss += iter_loss

                    t.set_postfix(loss=iter_loss)
                    t.update()

        mean_epoch_loss = epoch_loss / len(data_loader)
        return mean_epoch_loss

    def _train_iteration(self, data, storer):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        storer: dict
            Dictionary in which to store important variables for vizualisation.
        """
        batch_size, channel, height, width = data.size()
        if not data.is_cuda:
            data = data.to(self.device)

        try:
            recon_batch, latent_dist, latent_sample = self.model(data)
            loss = self.loss_f(data, recon_batch, latent_dist, self.model.training,
                               storer, latent_sample=latent_sample)
            self.optimizer.zero_grad()
            try:
                loss.backward()
                self.optimizer.step()
            except RuntimeError:
                # when updating lambda, loss has no grad information
                pass

        except ValueError:
            # for losses that use multiple optimizers (e.g. Factor)
            loss = self.loss_f.call_optimize(data, self.model, self.optimizer, storer)

        return loss.item()


class LossesLogger(object):
    """Class definition for objects to write data to log files in a
    form which is then easy to be plotted.
    """

    def __init__(self, file_path_name):
        """ Create a logger to store information for plotting. """
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        self.logger = logging.getLogger("losses_logger")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)

        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

        # store filename for batch history
        self._fname = file_path_name

    def log(self, epoch, losses_storer):
        """Write to the log file """
        for k, v in losses_storer.items():
            log_string = ",".join(str(item) for item in [epoch, k, mean(v)])
            self.logger.debug(log_string)

        # dump batch history
        batch_values = np.array(list(losses_storer.values())).T
        header = ' '.join(k for k in losses_storer.keys())
        fname = self._fname[:-4] + f'_epoch{epoch}.log'
        np.savetxt(fname, batch_values, header=header)


# HELPERS
def mean(l):
    """Compute the mean of a list"""
    return sum(l) / len(l)
