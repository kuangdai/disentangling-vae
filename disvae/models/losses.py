"""
Module containing all vae losses.
"""
import abc
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

from .discriminator import Discriminator
from disvae.utils.math import (log_density_gaussian, log_importance_weight_matrix,
                               matrix_log_density_gaussian)


LOSSES = ["VAE", "betaH", "betaB", "factor", "btcvae", "epsvae", "mepsvae"]
RECON_DIST = ["bernoulli", "laplace", "gaussian"]


# TO-DO: clean n_data and device
def get_loss_f(loss_name, **kwargs_parse):
    """Return the correct loss function given the argparse arguments."""
    kwargs_all = dict(record_loss_every=kwargs_parse["record_loss_every"],
                      rec_dist=kwargs_parse["rec_dist"],
                      steps_anneal=kwargs_parse["reg_anneal"])
    if loss_name == "betaH":
        return BetaHLoss(beta=kwargs_parse["betaH_B"], **kwargs_all)
    elif loss_name == "VAE":
        return BetaHLoss(beta=1, **kwargs_all)
    elif loss_name == "betaB":
        return BetaBLoss(C_init=kwargs_parse["betaB_initC"],
                         C_fin=kwargs_parse["betaB_finC"],
                         gamma=kwargs_parse["betaB_G"],
                         **kwargs_all)
    elif loss_name == "factor":
        return FactorKLoss(kwargs_parse["device"],
                           gamma=kwargs_parse["factor_G"],
                           disc_kwargs=dict(latent_dim=kwargs_parse["latent_dim"]),
                           optim_kwargs=dict(lr=kwargs_parse["lr_disc"], betas=(0.5, 0.9)),
                           **kwargs_all)
    elif loss_name == "btcvae":
        return BtcvaeLoss(kwargs_parse["n_data"],
                          alpha=kwargs_parse["btcvae_A"],
                          beta=kwargs_parse["btcvae_B"],
                          gamma=kwargs_parse["btcvae_G"],
                          **kwargs_all)
    elif loss_name == "epsvae":
        return EpsilonLoss(eps_recon=kwargs_parse['epsvae_constrain_reconstruction'],
                           eps=kwargs_parse['epsvae_epsilon'],
                           warmup=kwargs_parse['epsvae_warmup'],
                           L0=kwargs_parse['epsvae_L0'],
                           incr_L=kwargs_parse['epsvae_incr_L'],
                           interval_incr_L=kwargs_parse['epsvae_interval_incr_L'],
                           lbd_lr0=kwargs_parse['epsvae_lambda_lr'],
                           **kwargs_all)
    elif loss_name == "mepsvae":
        return MultiEpsilonLoss(kwargs_parse["n_data"],
                                eps_alpha=kwargs_parse['mepsvae_epsilon_alpha'],
                                eps_beta=kwargs_parse['mepsvae_epsilon_beta'],
                                eps_gamma=kwargs_parse['mepsvae_epsilon_gamma'],
                                warmup=kwargs_parse['mepsvae_warmup'],
                                L0=kwargs_parse['mepsvae_L0'],
                                incr_L=kwargs_parse['mepsvae_incr_L'],
                                interval_incr_L=kwargs_parse['mepsvae_interval_incr_L'],
                                lbd_lr0=kwargs_parse['mepsvae_lambda_lr'],
                                **kwargs_all)
    else:
        assert loss_name not in LOSSES
        raise ValueError("Uknown loss : {}".format(loss_name))


class BaseLoss(abc.ABC):
    """
    Base class for losses.

    Parameters
    ----------
    record_loss_every: int, optional
        Every how many steps to recorsd the loss.

    rec_dist: {"bernoulli", "gaussian", "laplace"}, optional
        Reconstruction distribution istribution of the likelihood on the each pixel.
        Implicitely defines the reconstruction loss. Bernoulli corresponds to a
        binary cross entropy (bse), Gaussian corresponds to MSE, Laplace
        corresponds to L1.

    steps_anneal: nool, optional
        Number of annealing steps where gradually adding the regularisation.
    """

    def __init__(self, record_loss_every=1, rec_dist="bernoulli", steps_anneal=0):
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every
        self.rec_dist = rec_dist
        self.steps_anneal = steps_anneal

    @abc.abstractmethod
    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Shape : (batch_size, n_chan,
            height, width).

        recon_data : torch.Tensor
            Reconstructed data. Shape : (batch_size, n_chan, height, width).

        latent_dist : tuple of torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).

        is_train : bool
            Whether currently in train mode.

        storer : dict
            Dictionary in which to store important variables for vizualisation.

        kwargs:
            Loss specific arguments
        """

    def _pre_call(self, is_train, storer):
        if is_train:
            self.n_train_steps += 1

        if not is_train or (self.n_train_steps - 1) % self.record_loss_every == 0:
            storer = storer
        else:
            storer = None

        return storer


class BetaHLoss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """

    def __init__(self, beta=4, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer)
        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if is_train else 1)
        loss = rec_loss + anneal_reg * (self.beta * kl_loss)

        if storer is not None:
            storer['loss'].append(loss.item())

        return loss


class BetaBLoss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    C_init : float, optional
        Starting annealed capacity C.

    C_fin : float, optional
        Final annealed capacity C.

    gamma : float, optional
        Weight of the KL divergence term.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Burgess, Christopher P., et al. "Understanding disentangling in
        $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
    """

    def __init__(self, C_init=0., C_fin=20., gamma=100., **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.C_init = C_init
        self.C_fin = C_fin

    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer)

        C = (linear_annealing(self.C_init, self.C_fin, self.n_train_steps, self.steps_anneal)
             if is_train else self.C_fin)

        loss = rec_loss + self.gamma * (kl_loss - C).abs()

        if storer is not None:
            storer['loss'].append(loss.item())

        return loss


class FactorKLoss(BaseLoss):
    """
    Compute the Factor-VAE loss as per Algorithm 2 of [1]

    Parameters
    ----------
    device : torch.device

    gamma : float, optional
        Weight of the TC loss term. `gamma` in the paper.

    discriminator : disvae.discriminator.Discriminator

    optimizer_d : torch.optim

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).
    """

    def __init__(self, device,
                 gamma=10.,
                 disc_kwargs={},
                 optim_kwargs=dict(lr=5e-5, betas=(0.5, 0.9)),
                 **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.device = device
        self.discriminator = Discriminator(**disc_kwargs).to(self.device)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), **optim_kwargs)

    def __call__(self, *args, **kwargs):
        raise ValueError("Use `call_optimize` to also train the discriminator")

    def call_optimize(self, data, model, optimizer, storer):
        storer = self._pre_call(model.training, storer)

        # factor-vae split data into two batches. In the paper they sample 2 batches
        batch_size = data.size(dim=0)
        half_batch_size = batch_size // 2
        data = data.split(half_batch_size)
        data1 = data[0]
        data2 = data[1]

        # Factor VAE Loss
        recon_batch, latent_dist, latent_sample1 = model(data1)
        rec_loss = _reconstruction_loss(data1, recon_batch,
                                        storer=storer,
                                        distribution=self.rec_dist)

        kl_loss = _kl_normal_loss(*latent_dist, storer)

        d_z = self.discriminator(latent_sample1)
        # We want log(p_true/p_false). If not using logisitc regression but softmax
        # then p_true = exp(logit_true) / Z; p_false = exp(logit_false) / Z
        # so log(p_true/p_false) = logit_true - logit_false
        tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()
        # with sigmoid (not good results) should be `tc_loss = (2 * d_z.flatten()).mean()`

        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if model.training else 1)
        vae_loss = rec_loss + kl_loss + anneal_reg * self.gamma * tc_loss

        if storer is not None:
            storer['loss'].append(vae_loss.item())
            storer['tc_loss'].append(tc_loss.item())

        if not model.training:
            # don't backprop if evaluating
            return vae_loss

        # Compute VAE gradients
        optimizer.zero_grad()
        vae_loss.backward(retain_graph=True)

        # Discriminator Loss
        # Get second sample of latent distribution
        latent_sample2 = model.sample_latent(data2)
        z_perm = _permute_dims(latent_sample2).detach()
        d_z_perm = self.discriminator(z_perm)

        # Calculate total correlation loss
        # for cross entropy the target is the index => need to be long and says
        # that it's first output for d_z and second for perm
        ones = torch.ones(half_batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros_like(ones)
        d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_perm, ones))
        # with sigmoid would be :
        # d_tc_loss = 0.5 * (self.bce(d_z.flatten(), ones) + self.bce(d_z_perm.flatten(), 1 - ones))

        # TO-DO: check ifshould also anneals discriminator if not becomes too good ???
        #d_tc_loss = anneal_reg * d_tc_loss

        # Compute discriminator gradients
        self.optimizer_d.zero_grad()
        d_tc_loss.backward()

        # Update at the end (since pytorch 1.5. complains if update before)
        optimizer.step()
        self.optimizer_d.step()

        if storer is not None:
            storer['discrim_loss'].append(d_tc_loss.item())

        return vae_loss


class BtcvaeLoss(BaseLoss):
    """
    Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]

    Parameters
    ----------
    n_data: int
        Number of data in the training set

    alpha : float
        Weight of the mutual information term.

    beta : float
        Weight of the total correlation term.

    gamma : float
        Weight of the dimension-wise KL term.

    is_mss : bool
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(self, n_data, alpha=1., beta=6., gamma=1., is_mss=True, **kwargs):
        super().__init__(**kwargs)
        self.n_data = n_data
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.is_mss = is_mss  # minibatch stratified sampling

    def __call__(self, data, recon_batch, latent_dist, is_train, storer,
                 latent_sample=None):
        storer = self._pre_call(is_train, storer)
        batch_size, latent_dim = latent_sample.shape

        rec_loss = _reconstruction_loss(data, recon_batch,
                                        storer=storer,
                                        distribution=self.rec_dist)
        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample,
                                                                             latent_dist,
                                                                             self.n_data,
                                                                             is_mss=self.is_mss)
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if is_train else 1)

        # total loss
        loss = rec_loss + (self.alpha * mi_loss +
                           self.beta * tc_loss +
                           anneal_reg * self.gamma * dw_kl_loss)

        if storer is not None:
            storer['loss'].append(loss.item())
            storer['mi_loss'].append(mi_loss.item())
            storer['tc_loss'].append(tc_loss.item())
            storer['dw_kl_loss'].append(dw_kl_loss.item())
            # computing this for storing and comparaison purposes
            _ = _kl_normal_loss(*latent_dist, storer)

        return loss


class EpsilonLoss(BaseLoss):
    """
    Epsilon loss, single constraint on reconstruction or KL
    """

    def __init__(self, eps_recon=False, eps=.1,
                 warmup=100, L0=1, incr_L=1, interval_incr_L=2,
                 lbd_lr0=0.01, **kwargs):
        super().__init__(**kwargs)
        # parameters
        self.eps_recon = eps_recon
        self.eps = eps
        self.warmup = warmup
        self.L0 = L0
        self.incr_L = incr_L
        self.interval_incr_L = interval_incr_L
        self.lbd_lr0 = lbd_lr0

        # hardcoded
        self.lbd_lr_decay_rate = 1e-4
        self.lbd_lr_decay_step = 1.

        # to be updated in training
        self.n_total_updates = 0
        self.n_lbd_updates = 0
        self.n_weight_updates_since_last_lbd_update = 0
        self.lbd = 0.
        self.L = self.L0

    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        storer = self._pre_call(is_train, storer)

        # training lambda flag
        training_lbd = self.n_weight_updates_since_last_lbd_update == self.L and \
                       self.n_total_updates >= self.warmup and is_train
        if storer is not None:
            storer['training_lambda'].append(training_lbd * 1.)

        ################
        # compute loss #
        ################
        # rec and KL
        if training_lbd:
            with torch.no_grad():
                # update lbd, so no_grad for weights
                rec_loss = _reconstruction_loss(data, recon_data,
                                                storer=storer,
                                                distribution=self.rec_dist)
                kl_loss = _kl_normal_loss(*latent_dist, storer)
        else:
            rec_loss = _reconstruction_loss(data, recon_data,
                                            storer=storer,
                                            distribution=self.rec_dist)
            kl_loss = _kl_normal_loss(*latent_dist, storer)
        # loss
        if self.eps_recon:
            loss = self.lbd * F.relu(rec_loss - self.eps) + kl_loss
        else:
            loss = self.lbd * F.relu(kl_loss - self.eps) + rec_loss
        if storer is not None:
            storer['loss'].append(loss.item())

        #################
        # update Lambda #
        #################
        # lbd learning rate
        lbd_lr = self.lbd_lr0 / (1 + self.lbd_lr_decay_rate *
                                 self.n_lbd_updates / self.lbd_lr_decay_step)
        if training_lbd:
            # update lda
            if self.eps_recon:
                self.lbd += F.relu(rec_loss - self.eps).item() * lbd_lr
            else:
                self.lbd += F.relu(kl_loss - self.eps).item() * lbd_lr

            # update lbd steps
            self.n_lbd_updates += 1

            # update L
            if self.n_lbd_updates % self.interval_incr_L == 0:
                self.L += self.incr_L

            # reset
            self.n_weight_updates_since_last_lbd_update = 0
        else:
            if self.n_total_updates >= self.warmup and is_train:
                self.n_weight_updates_since_last_lbd_update += 1

        # record
        if storer is not None:
            storer['lambda'].append(self.lbd)
            storer['lambda_lr'].append(lbd_lr)

        # must be placed here
        if is_train:
            self.n_total_updates += 1
        return loss


class MultiEpsilonLoss(BaseLoss):
    """
    Epsilon loss, multiple constraints on alpha, beta, gamma terms
    """
    def __init__(self, n_data, eps_alpha=.1, eps_beta=.1, eps_gamma=.1,
                 is_mss=True, warmup=100, L0=1, incr_L=1, interval_incr_L=2,
                 lbd_lr0=0.01, **kwargs):
        super().__init__(**kwargs)
        # parameters
        self.n_data = n_data
        self.eps_alpha = eps_alpha
        self.eps_beta = eps_beta
        self.eps_gamma = eps_gamma
        self.is_mss = is_mss
        self.warmup = warmup
        self.L0 = L0
        self.incr_L = incr_L
        self.interval_incr_L = interval_incr_L
        self.lbd_lr0 = lbd_lr0

        # hardcoded
        self.lbd_lr_decay_rate = 1e-4
        self.lbd_lr_decay_step = 1.

        # to be updated in training
        self.n_total_updates = 0
        self.n_lbd_updates = 0
        self.n_weight_updates_since_last_lbd_update = 0
        self.lbd_alpha = 0.
        self.lbd_beta = 0.
        self.lbd_gamma = 0.
        self.L = self.L0

    def __call__(self, data, recon_data, latent_dist, is_train, storer,
                 latent_sample=None):
        storer = self._pre_call(is_train, storer)

        # training lambda flag
        training_lbd = self.n_weight_updates_since_last_lbd_update == self.L and \
                       self.n_total_updates >= self.warmup and is_train
        if storer is not None:
            storer['training_lambda'].append(training_lbd * 1.)

        ################
        # compute loss #
        ################
        # rec and KL
        if training_lbd:
            with torch.no_grad():
                # update lbd, so no_grad for weights
                rec_loss = _reconstruction_loss(data, recon_data,
                                                storer=storer,
                                                distribution=self.rec_dist)
                log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(
                    latent_sample,
                    latent_dist,
                    self.n_data,
                    is_mss=self.is_mss)
                mi_loss = (log_q_zCx - log_qz).mean()
                tc_loss = (log_qz - log_prod_qzi).mean()
                dw_kl_loss = (log_prod_qzi - log_pz).mean()
        else:
            rec_loss = _reconstruction_loss(data, recon_data,
                                            storer=storer,
                                            distribution=self.rec_dist)
            log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(
                latent_sample,
                latent_dist,
                self.n_data,
                is_mss=self.is_mss)
            mi_loss = (log_q_zCx - log_qz).mean()
            tc_loss = (log_qz - log_prod_qzi).mean()
            dw_kl_loss = (log_prod_qzi - log_pz).mean()

        # loss
        loss = rec_loss + (self.lbd_alpha * F.relu(mi_loss - self.eps_alpha) +
                           self.lbd_beta * F.relu(tc_loss - self.eps_beta) +
                           self.lbd_gamma * F.relu(dw_kl_loss - self.eps_gamma))
        if storer is not None:
            storer['loss'].append(loss.item())
            storer['mi_loss'].append(mi_loss.item())
            storer['tc_loss'].append(tc_loss.item())
            storer['dw_kl_loss'].append(dw_kl_loss.item())
            # computing this for storing and comparaison purposes
            _ = _kl_normal_loss(*latent_dist, storer)

        #################
        # update Lambda #
        #################
        # lbd learning rate
        lbd_lr = self.lbd_lr0 / (1 + self.lbd_lr_decay_rate *
                                 self.n_lbd_updates / self.lbd_lr_decay_step)
        if training_lbd:
            # update lda
            self.lbd_alpha += F.relu(mi_loss - self.eps_alpha).item() * lbd_lr
            self.lbd_beta += F.relu(tc_loss - self.eps_beta).item() * lbd_lr
            self.lbd_gamma += F.relu(dw_kl_loss - self.eps_gamma).item() * lbd_lr

            # update lbd steps
            self.n_lbd_updates += 1

            # update L
            if self.n_lbd_updates % self.interval_incr_L == 0:
                self.L += self.incr_L

            # reset
            self.n_weight_updates_since_last_lbd_update = 0
        else:
            if self.n_total_updates >= self.warmup and is_train:
                self.n_weight_updates_since_last_lbd_update += 1

        # record
        if storer is not None:
            storer['lambda_alpha'].append(self.lbd_alpha)
            storer['lambda_beta'].append(self.lbd_beta)
            storer['lambda_gamma'].append(self.lbd_gamma)
            storer['lambda_lr'].append(lbd_lr)

        # must be placed here
        if is_train:
            self.n_total_updates += 1
        return loss


def _reconstruction_loss(data, recon_data, distribution="bernoulli", storer=None):
    """
    Calculates the per image reconstruction loss for a batch of data. I.e. negative
    log likelihood.

    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Shape : (batch_size, n_chan,
        height, width).

    recon_data : torch.Tensor
        Reconstructed data. Shape : (batch_size, n_chan, height, width).

    distribution : {"bernoulli", "gaussian", "laplace"}
        Distribution of the likelihood on the each pixel. Implicitely defines the
        loss Bernoulli corresponds to a binary cross entropy (bse) loss and is the
        most commonly used. It has the issue that it doesn't penalize the same
        way (0.1,0.2) and (0.4,0.5), which might not be optimal. Gaussian
        distribution corresponds to MSE, and is sometimes used, but hard to train
        ecause it ends up focusing only a few pixels that are very wrong. Laplace
        distribution corresponds to L1 solves partially the issue of MSE.

    storer : dict
        Dictionary in which to store important variables for vizualisation.

    Returns
    -------
    loss : torch.Tensor
        Per image cross entropy (i.e. normalized per batch but not pixel and
        channel)
    """
    batch_size, n_chan, height, width = recon_data.size()
    is_colored = n_chan == 3

    if distribution == "bernoulli":
        loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
    elif distribution == "gaussian":
        # loss in [0,255] space but normalized by 255 to not be too big
        loss = F.mse_loss(recon_data * 255, data * 255, reduction="sum") / 255
    elif distribution == "laplace":
        # loss in [0,255] space but normalized by 255 to not be too big but
        # multiply by 255 and divide 255, is the same as not doing anything for L1
        loss = F.l1_loss(recon_data, data, reduction="sum")
        loss = loss * 3  # emperical value to give similar values than bernoulli => use same hyperparam
        loss = loss * (loss != 0)  # masking to avoid nan
    else:
        assert distribution not in RECON_DIST
        raise ValueError("Unkown distribution: {}".format(distribution))

    loss = loss / batch_size

    if storer is not None:
        storer['recon_loss'].append(loss.item())

    return loss


def _kl_normal_loss(mean, logvar, storer=None):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim) where
        D is dimension of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)

    storer : dict
        Dictionary in which to store important variables for vizualisation.
    """
    latent_dim = mean.size(1)
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    if storer is not None:
        storer['kl_loss'].append(total_kl.item())
        for i in range(latent_dim):
            storer['kl_loss_' + str(i)].append(latent_kl[i].item())

    return total_kl


def _permute_dims(latent_sample):
    """
    Implementation of Algorithm 1 in ref [1]. Randomly permutes the sample from
    q(z) (latent_dist) across the batch for each of the latent dimensions (mean
    and log_var).

    Parameters
    ----------
    latent_sample: torch.Tensor
        sample from the latent dimension using the reparameterisation trick
        shape : (batch_size, latent_dim).

    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).

    """
    perm = torch.zeros_like(latent_sample)
    batch_size, dim_z = perm.size()

    for z in range(dim_z):
        pi = torch.randperm(batch_size).to(latent_sample.device)
        perm[:, z] = latent_sample[pi, z]

    return perm


def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data,
                               is_mss=True):
    batch_size, hidden_dim = latent_sample.shape

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    if not is_mss:
        log_qz, log_prod_qzi = _minibatch_weighted_sampling(latent_dist,
                                                            latent_sample,
                                                            n_data)

    else:
        log_qz, log_prod_qzi = _minibatch_stratified_sampling(latent_dist,
                                                              latent_sample,
                                                              n_data)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx


def _minibatch_weighted_sampling(latent_dist, latent_sample, data_size):
    """
    Estimates log q(z) and the log (product of marginals of q(z_j)) with minibatch
    weighted sampling.

    Parameters
    ----------
    latent_dist : tuple of torch.tensor
        sufficient statistics of the latent dimension. E.g. for gaussian
        (mean, log_var) each of shape : (batch_size, latent_dim).

    latent_sample: torch.Tensor
        sample from the latent dimension using the reparameterisation trick
        shape : (batch_size, latent_dim).

    data_size : int
        Number of data in the training set

    References
    -----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """
    batch_size = latent_sample.size(0)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    log_prod_qzi = (torch.logsumexp(mat_log_qz, dim=1, keepdim=False) -
                    math.log(batch_size * data_size)).sum(dim=1)
    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False
                             ) - math.log(batch_size * data_size)

    return log_qz, log_prod_qzi


def _minibatch_stratified_sampling(latent_dist, latent_sample, data_size):
    """
    Estimates log q(z) and the log (product of marginals of q(z_j)) with minibatch
    stratified sampling.

    Parameters
    -----------
    latent_dist : tuple of torch.tensor
        sufficient statistics of the latent dimension. E.g. for gaussian
        (mean, log_var) each of shape : (batch_size, latent_dim).

    latent_sample: torch.Tensor
        sample from the latent dimension using the reparameterisation trick
        shape : (batch_size, latent_dim).

    data_size : int
        Number of data in the training set

    References
    -----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """
    batch_size = latent_sample.size(0)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    log_iw_mat = log_importance_weight_matrix(batch_size, data_size).to(
        latent_sample.device)
    log_qz = torch.logsumexp(log_iw_mat + mat_log_qz.sum(2), dim=1,
                             keepdim=False)
    log_prod_qzi = torch.logsumexp(log_iw_mat.view(batch_size, batch_size, 1) +
                                   mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_qz, log_prod_qzi

