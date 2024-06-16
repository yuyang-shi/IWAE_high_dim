import torch
from torch import nn
from torch.distributions import Normal, Bernoulli, Independent
import pytorch_lightning as pl
import numpy as np

import utils


def get_nonlinearity(nonlinearity):
    if nonlinearity == 'tanh':
        return nn.Tanh()


def MLP(net_dims, nonlinearity):
    model = nn.Sequential()
    for in_dim, out_dim in zip(net_dims[:-2], net_dims[1:-1]):
        model.append(nn.Linear(in_dim, out_dim))
        model.append(nonlinearity)
    model.append(nn.Linear(net_dims[-2], net_dims[-1]))
    return model


class MLPVAE(pl.LightningModule):
    def __init__(self, dual_optimizers, loss_fn, test_fn, args):
        super().__init__()
        self.args = args
        self.latent_dim = args.latent_dim
        self.obs_dim = args.obs_dim
        assert len(args.obs_dim) == 1, "obs_dim should have length 1 in MLP VAE"
        self.loss_fn = loss_fn
        self.test_fn = test_fn

        nonlinearity = get_nonlinearity(args.model.nonlinearity)
        self.decoder = MLP([self.latent_dim] + list(args.model.net_dims) + [self.decoder_out_dim], nonlinearity)
        self.encoder = MLP([self.obs_dim[0]] + list(args.model.net_dims) + [self.latent_dim*2], nonlinearity)

        self.register_buffer("prior_mu", torch.zeros(self.latent_dim))
        self.register_buffer("prior_sig", torch.ones(self.latent_dim))

        self.dual_optimizers = dual_optimizers
        self.automatic_optimization = not dual_optimizers

    def forward(self, x):
        # reconstruct x
        qz_x = self.qz_x_fn(x)
        z_recon = qz_x.sample()
        x_recon = self.px_z_fn(z_recon).sample()
        return x_recon, z_recon

    def sample(self, n):
        z_uncon = self.pz.sample((n,))
        x_uncon = self.px_z_fn(z_uncon).sample()
        return x_uncon, z_uncon

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch

        if self.dual_optimizers:
            optimizer_theta, optimizer_phi = self.optimizers()
            loss_theta, loss_phi, log_ws = self.loss_fn(x, self.pz, self.px_z_fn, self.qz_x_fn,
                                                    encoder=self.encoder, qz_x_dist_fn=self.qz_x_dist_fn)

            optimizer_phi.zero_grad()
            self.manual_backward(loss_phi, retain_graph=True)
            optimizer_theta.zero_grad()
            self.manual_backward(loss_theta)
            optimizer_theta.step()
            optimizer_phi.step()

            loss = loss_theta
        else:
            loss, log_ws = self.loss_fn(x, self.pz, self.px_z_fn, self.qz_x_fn,
                                        encoder=self.encoder, qz_x_dist_fn=self.qz_x_dist_fn)

        self.log("train/loss", loss)
        self.log("train/ess", utils.compute_ess(log_ws))
        self.log("train/kss", utils.compute_kss(log_ws))
        self.log("train/B_d", utils.compute_B_d_from_log_ws(log_ws))
        self.log("train/stdev_log_ws", utils.compute_stdev_from_log_ws(log_ws))
        self.log("train/gamma_alpha", utils.compute_gamma_alpha_from_log_ws(log_ws, alpha=self.args.alpha))
        if not self.dual_optimizers:
            return loss

    def configure_optimizers(self):
        if self.dual_optimizers:
            optimizer_theta = utils.get_optimizer(self.decoder.parameters(), self.args)
            optimizer_phi = utils.get_optimizer(self.encoder.parameters(), self.args)
            return optimizer_theta, optimizer_phi
        else:
            optimizer = utils.get_optimizer(self.parameters(), self.args)
            return optimizer

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, _ = batch
        valid_loss, log_ws = self.test_fn(x, self.pz, self.px_z_fn, self.qz_x_fn)
        self.log(f"valid/iwae{self.args.test_N}_loss", valid_loss)
        self.log("valid/ess", utils.compute_ess(log_ws))
        self.log("valid/kss", utils.compute_kss(log_ws))
        self.log("valid/B_d", utils.compute_B_d_from_log_ws(log_ws))
        self.log("valid/stdev_log_ws", utils.compute_stdev_from_log_ws(log_ws))
        self.log("valid/gamma_alpha", utils.compute_gamma_alpha_from_log_ws(log_ws, alpha=self.args.alpha))

        if batch_idx == 1:
            self.plot(x, 'valid')

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, _ = batch
        test_loss, log_ws = self.test_fn(x, self.pz, self.px_z_fn, self.qz_x_fn)
        self.log(f"test/iwae{self.args.test_N}_loss", test_loss)
        self.log("test/ess", utils.compute_ess(log_ws))
        self.log("test/kss", utils.compute_kss(log_ws))
        self.log("test/B_d", utils.compute_B_d_from_log_ws(log_ws))
        self.log("test/stdev_log_ws", utils.compute_stdev_from_log_ws(log_ws))
        self.log("test/gamma_alpha", utils.compute_gamma_alpha_from_log_ws(log_ws, alpha=self.args.alpha))

        if batch_idx == 1:
            self.plot(x, 'test')

    @torch.no_grad()
    def plot(self, x, dl_name='test'):
        x_recon, z_recon = self(x)
        x_uncon, z_uncon = self.sample(x.shape[0])

        if self.args.data.type == 'image':
            nrow = utils.factor_int(x.shape[0])[1]
            self.logger.log_image(dl_name + "/im_grid_truth", [utils.to_pil_image(x, self.args, nrow=nrow)])
            self.logger.log_image(dl_name + "/im_grid_recon", [utils.to_pil_image(x_recon, self.args, nrow=nrow)])
            self.logger.log_image(dl_name + "/im_grid_uncon", [utils.to_pil_image(x_uncon, self.args, nrow=nrow)])

    def px_z_fn(self, z):
        return self.px_z_dist_fn(self.decoder(z))

    def qz_x_fn(self, x):
        return self.qz_x_dist_fn(self.encoder(x))

    def px_z_dist_fn(self, decode):
        if self.args.model.distribution_type == 'Bernoulli':
            return Independent(Bernoulli(logits=decode), 1)
        elif self.args.model.distribution_type == 'Normal':
            mu, log_sig = decode.chunk(2, dim=-1)
            return Independent(Normal(mu, log_sig.exp()), 1)

    def qz_x_dist_fn(self, encode):
        mu, log_sig = encode.chunk(2, dim=-1)
        return Independent(Normal(mu, log_sig.exp()), 1)

    @property
    def pz(self):
        return Independent(Normal(self.prior_mu, self.prior_sig), 1)

    @property
    def decoder_out_dim(self):
        if self.args.model.distribution_type == 'Bernoulli':
            return self.obs_dim[0]
        elif self.args.model.distribution_type == 'Normal':
            return self.obs_dim[0]*2
