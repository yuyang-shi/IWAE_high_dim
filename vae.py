import os
from functools import partial
import pytorch_lightning as pl
import hydra

import utils
import losses
from datasets import load_dataset
from models import MLPVAE


@hydra.main(config_path="./conf", config_name="vae")
def main(args):
    seed = args.seed
    N = args.N
    loss_name = args.loss_name
    alpha = args.alpha

    if seed is not None:
        utils.manual_seed(seed)

    dual_optimizers = False
    loss_fn = partial(getattr(losses, loss_name), N=N, alpha=alpha)
    if loss_name in ['alpha_iwae_dreg_loss_v3', 'alpha_iwae_dreg_loss_v4']:
        dual_optimizers = True
    test_fn = partial(losses.vr_iwae_loss_v1, N=5000)

    # init the autoencoder
    if args.checkpoint_path is None:
        vae = MLPVAE(dual_optimizers, loss_fn, test_fn, args)
    else:
        print("Resume using checkpoint", args.checkpoint_path)
        vae = MLPVAE.load_from_checkpoint(checkpoint_path=hydra.utils.to_absolute_path(args.checkpoint_path),
                                          dual_optimizers=dual_optimizers, loss_fn=loss_fn, test_fn=test_fn, args=args)

    # setup data
    train_loader, valid_loader, test_loader = load_dataset(args)
    logger = utils.get_logger(args)
    checkpoint_callback = utils.get_checkpoint_callback(args)

    # initial save and test
    trainer = pl.Trainer(logger=logger, gpus=1, limit_train_batches=0)
    trainer.fit(vae, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.validate(vae, valid_loader)
    trainer.test(vae, test_loader)
    os.mkdir('checkpoints')
    trainer.save_checkpoint('checkpoints/epoch=0-step=0.ckpt')

    # train the model
    trainer = pl.Trainer(logger=logger, callbacks=[checkpoint_callback], gpus=1, max_epochs=args.max_epochs, check_val_every_n_epoch=args.check_val_every_n_epoch)
    trainer.fit(vae, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.test(vae, test_loader)


if __name__ == '__main__':
    main()