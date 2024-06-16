import os
import random
import numpy as np
import arviz
import torch
import torchvision.utils as vutils
import hydra
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning.loggers import CSVLogger as _CSVLogger, WandbLogger
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def manual_seed(seed=0):
    print("Set seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimizer(params, args):
    lr = args.lr
    optimizer = args.optimizer
    if optimizer == 'SGD':
        return torch.optim.SGD(params, lr)
    elif optimizer == 'Adam':
        return torch.optim.Adam(params, lr=lr)


def get_logger(args):
    if args.nosave:
        logger = DummyLogger()
    elif args.logger == 'CSV':
        class CSVLogger(_CSVLogger):
            def log_image(self, key, images, **kwargs):
                pass

        logger = CSVLogger(os.getcwd())
    elif args.logger == 'Wandb':
        run_name = os.path.normpath(os.path.relpath(os.getcwd(), os.path.join(
            hydra.utils.to_absolute_path(args.paths.experiments_dir_name), args.name))).replace("\\", "/")
        config = OmegaConf.to_container(args, resolve=True)
        kwargs = {'name': run_name, 'project': 'IWAEhighdim_' + args.name, 'entity': "yuyshi-team", "config": config}
        logger = WandbLogger(**kwargs)
    else:
        raise NotImplementedError
    return logger


def get_checkpoint_callback(args):
    if args.checkpoint_every_n_epochs == -1:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), "checkpoints"),
            save_last=True
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), "checkpoints"),
            save_last=True,
            save_top_k=-1,
            every_n_epochs=args.checkpoint_every_n_epochs,
            save_on_train_epoch_end=True
        )
    return checkpoint_callback


def lognormexp(values, dim):
    log_denom = values.logsumexp(dim=dim, keepdim=True)
    return values - log_denom


def logmeanexp(values, dim):
    return values.logsumexp(dim=dim) - np.log(values.shape[dim])


def compute_ess(log_ws, dim=0):
    log_numer = log_ws.logsumexp(dim=dim) * 2
    log_denom = (log_ws * 2).logsumexp(dim=dim)
    return (log_numer - log_denom).exp().mean()


def compute_kss(log_ws, dim=0):
    dims = list(range(len(log_ws.shape)))
    dims.remove(dim)
    dims = dims + [dim]
    _, khats = arviz.psislw(log_ws.permute(dims).detach().cpu().numpy())
    return khats.mean()


def compute_B_d_from_log_ws(log_ws, dim=0):
    log_ws = log_ws - logmeanexp(log_ws, dim=dim)
    mus = log_ws.mean(dim=dim)
    B_ds = (-2 * mus).sqrt()
    return B_ds.mean()


def compute_stdev_from_log_ws(log_ws, dim=0):
    # Returns the standard deviation of the log weights
    return torch.std(log_ws, dim=dim).mean()


def compute_gamma_alpha_from_log_ws(log_ws, alpha=0.0, dim=0):
    # Returns gamma_alpha^2 calculated using weight samples
    alpha_log_ws = (1-alpha) * log_ws
    alpha_log_ws = alpha_log_ws - logmeanexp(alpha_log_ws, dim=dim)
    gammas = torch.var(alpha_log_ws.exp(), dim=dim) / (1-alpha)
    return gammas.mean()


def factor_int(n):
    val = np.ceil(np.sqrt(n))
    val2 = int(n/val)
    while val2 * val != float(n):
        val -= 1
        val2 = int(n/val)
    return int(val), val2  # smaller, larger


@torch.no_grad()
def to_pil_image(tensor, args, **kwargs):
    tensor = tensor.view((tensor.shape[0], args.data.channels, args.data.image_size, args.data.image_size))
    grid = vutils.make_grid(tensor, **kwargs)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


@torch.no_grad()
def save_image(tensor, args, fp, format=None, **kwargs):
    im = to_pil_image(tensor, args, **kwargs)
    im.save(fp, format=format)