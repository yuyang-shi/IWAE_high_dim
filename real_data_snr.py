import os
from functools import partial
import hydra
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

import losses
from datasets import load_dataset
from utils import logmeanexp, lognormexp, manual_seed, get_optimizer
from models import MLPVAE
from log_normal_gaussian import compute_gamma_alpha_LogNormal, compute_gap_1overN, compute_gap_LogNormal
from linear_gaussian_snr import compute_gap_approx_LogNormal


def get_model_for_plotting(args, dim=None, alpha=None):
    # Reads config and returns a model for plotting, plus test_loader
    # If checkpoint is passed, uses checkpointed model
    # Else, creates a new untrained model from scratch

    seed = args.seed
    N = args.N
    loss_name = args.loss_name

    if alpha is not None:
        if args.checkpoint_path is not None:
            raise ValueError('Trying to customise latent dim while loading checkpoint')
        args.alpha = alpha

    if seed is not None:
        manual_seed(seed)

    dual_optimizers = False
    loss_fn = partial(getattr(losses, loss_name), N=N, alpha=alpha)
    if loss_name in ['alpha_iwae_dreg_loss_v3', 'alpha_iwae_dreg_loss_v4']:
        dual_optimizers = True
    test_fn = partial(losses.vr_iwae_loss_v1, N=5000)

    # customise the latent dimension (only possible if not loading from checkpoint)
    if dim:
        if args.checkpoint_path is not None:
            raise ValueError('Trying to customise latent dim while loading checkpoint')
        args.latent_dim = dim

    # init the autoencoder
    if args.checkpoint_path is None:
        vae = MLPVAE(dual_optimizers, loss_fn, test_fn, args)
    else:
        print("Resume using checkpoint", args.checkpoint_path)
        vae = MLPVAE.load_from_checkpoint(checkpoint_path=hydra.utils.to_absolute_path(args.checkpoint_path),
                                          dual_optimizers=dual_optimizers, loss_fn=loss_fn, test_fn=test_fn, args=args)

    # setup data, logging and model
    _, _, test_loader = load_dataset(args)

    return vae, test_loader


def get_log_weights(model, x, N):
    # Returns N weight samples from the model with input x
    x = x.view(x.size(0), *model.obs_dim)[0].unsqueeze(dim=0)
    _, log_ws = losses.vr_iwae_loss_v1(x, model.pz, model.px_z_fn, model.qz_x_fn, N=N)
    return log_ws


@torch.no_grad()
def log_ws_plots_and_computations(model, x, args, figure_path, dim, i, alpha):
    # Plots the distribution of the log weights
    # Returns estimated mean and std of associated normal distribution
    # as well as an estimation of gamma_alpha

    fig, axes = plt.subplots()
    log_ws = get_log_weights(model, x, args.weight_samples)
    logmeanexp_log_ws = logmeanexp(log_ws, dim=0)

    alpha_log_ws = (1-alpha) * log_ws
    logmeanexp_alpha_log_ws = logmeanexp(alpha_log_ws, dim=0)
    alpha_log_ws = alpha_log_ws - logmeanexp_alpha_log_ws
    gamma = torch.var(alpha_log_ws.exp()) / (1-alpha)

    baseline_VR_bound_estim = logmeanexp_alpha_log_ws.item() / (1-alpha)
    baseline_log_lik_estim = logmeanexp_log_ws.item()

    log_ws = log_ws - logmeanexp_log_ws
    log_ws = log_ws.flatten().detach().cpu().numpy()

    values, bins = np.histogram(log_ws, bins=81, density=True)
    bins = (bins[1:] + bins[:-1]) / 2
    axes.plot(bins, values)

    axes.set_xlabel(r"$\log \overline{w}_i$")
    axes.set_ylabel("Density")
    plt.title(r"Distribution of $\log \overline{w}_i$ ($d =$" + str(dim) +")")
    plt.savefig(figure_path + '/weight_distr_'+ str(dim) + 'datapoints' + str(i) +'.png')
    plt.close()

    qqplot(log_ws, line='45', fit=True)
    plt.title(r"QQ-plot ($d=$" + str(dim) + ")")
    plt.savefig(figure_path + '/qqplot_'+ str(dim) + 'datapoints' + str(i) +'.png')
    plt.close()

    mu, std = norm.fit(log_ws)

    return mu, std, gamma.detach().cpu().numpy(), baseline_VR_bound_estim, baseline_log_lik_estim


@torch.no_grad()
def compute_vr_iwae_bound(model, args, J, x, alpha):
    x = x.view(x.size(0), *model.obs_dim)[0].unsqueeze(dim=0)

    vr_iwae_lst = []

    for j in range(1, J):
        loss, _ = losses.vr_iwae_loss_v1(x, model.pz, model.px_z_fn, model.qz_x_fn, N=2**j, alpha=alpha)
        vr_iwae_lst.append(-loss.detach().cpu().numpy())

    return vr_iwae_lst


# @torch.no_grad()
# def baseline_VR_bound(x, model, nb_samples=10000, alpha=0.0, nb_repeat=50):
#     baselines = []
#
#     x = x.view(x.size(0), *model.obs_dim)[0].unsqueeze(dim=0)
#
#     for _ in range(nb_repeat):
#         loss, _ = losses.vr_iwae_loss_v1(x, model.pz, model.px_z_fn, model.qz_x_fn, N=nb_samples, alpha=alpha)
#         baselines.append(loss.detach().cpu().numpy())
#     baseline = sum(baselines) / len(baselines)
#
#     return baseline


@torch.no_grad()
def compute_vr_bound(model, x, N, alpha):
    # Returns the VR objective on datapoint x calculated with N weights samples
    log_ws = get_log_weights(model, x, N)
    bound = logmeanexp((1-alpha) * log_ws, dim=0) / (1-alpha)
    return bound.detach().numpy()


@torch.no_grad()
def plot_vr_bound(model, test_loader, args, figure_path):
    # Plots the VR bound, calculated with N weights samples, against alpha
    test_loader_iter = iter(test_loader)
    alphas = np.linspace(0,0.9,10)
    fig, axes = plt.subplots()

    for _ in range(args.datapoints):
        bounds = []
        x, _ = next(test_loader_iter)

        for alpha in alphas:
            bounds.append(compute_vr_bound(model, x, args.weight_samples, alpha))

        axes.plot(alphas, bounds)

    axes.set_xlabel(r"$\alpha$")
    axes.set_ylabel(r"$\mathcal{L}^{(\alpha)}$")
    plt.savefig(figure_path + '/vr_bound_against_alpha.png')


def compute_snr_results(model, loss_name, data, alpha, args, nb_repeat=10**4):
    optimizer = get_optimizer(model.parameters(), args)

    p_grad_idx = np.rint(np.linspace(0, len(nn.utils.parameters_to_vector(model.decoder.parameters()))-1, 10)).astype(int)
    q_grad_idx = np.rint(np.linspace(0, len(nn.utils.parameters_to_vector(model.encoder.parameters()))-1, 10)).astype(int)

    J = 13
    results_keys = ["vr_iwae_snr", "vr_iwae_p_grad_snr", "vr_iwae_q_grad_snr"]
    results = {key: [] for key in results_keys}

    for j in tqdm(range(1, J)):
        results_repeat_keys = ["vr_iwae", "vr_iwae_p_grad", "vr_iwae_q_grad"]
        results_repeat = {key: [] for key in results_repeat_keys}

        for _ in range(nb_repeat):
            optimizer.zero_grad()
            loss = getattr(losses, loss_name)(data, model.pz, model.px_z_fn, model.qz_x_fn, N=2 ** j, alpha=alpha,
                                              encoder=model.encoder, qz_x_dist_fn=model.qz_x_dist_fn)[0]
            vr_iwae = -loss
            vr_iwae.backward()

            results_repeat["vr_iwae"].append(vr_iwae.item())
            results_repeat["vr_iwae_p_grad"].append(nn.utils.parameters_to_vector([param.grad for param in model.decoder.parameters()])[p_grad_idx].detach().cpu().numpy())
            results_repeat["vr_iwae_q_grad"].append(nn.utils.parameters_to_vector([param.grad for param in model.encoder.parameters()])[q_grad_idx].detach().cpu().numpy())

        for key in results_repeat_keys:
            results[f"{key}_snr"].append(np.nanmean(np.abs(np.mean(results_repeat[key], axis=0)) / np.std(results_repeat[key], axis=0)))

    df_results = pd.DataFrame(results)
    return df_results


@hydra.main(config_path="./conf", config_name="weights")
def main(args):
    # Plot settings
    plt.style.use('bmh')
    plt.rcParams['figure.facecolor'] = '1'
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams["legend.loc"] = 'lower right'

    # Make subdirectory of ./figure/ for run
    figure_dir = hydra.utils.to_absolute_path('figures_real_data')
    run_name = os.path.normpath(os.path.relpath(os.getcwd(), os.path.join(
            hydra.utils.to_absolute_path(args.paths.experiments_dir_name), args.name))).replace("\\", "/")
    figure_path = figure_dir + '/' + run_name

    try:
        os.makedirs(figure_path)
    except OSError:
        pass

    # Load model and data
    dim_list = [1, 5, 10, 20, 50, 100, 1000, 2500, 5000, 10000]
    alpha_list = [0, 0.2, 0.5, 0.8, 1]
    loss_name_list = ["vr_iwae_dreg_loss_v6"]
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(device)

    J = 13
    plt_iters = [2 ** j for j in range(1, J)]
    nb_runs = 100
    # nb_samples_baseline = 2000
    epoch = 0

    for loss_name in loss_name_list:
        rep_name =  "rep" if loss_name == "vr_iwae_loss_v2" else "drep"
        for dim in dim_list:
            df_results_all = []
            for alpha in alpha_list:
                vae, test_loader = get_model_for_plotting(args, dim, alpha)
                vae = vae.to(device)
                test_loader_iter = iter(test_loader)

                # for i in range(args.datapoints):
                x, _ = next(test_loader_iter)
                x = x.to(device)

                df_results = compute_snr_results(vae, loss_name, x[0:1], alpha, args)
                df_results.index = 2 ** (df_results.index + 1)
                df_results_all.append(df_results)

            df_results_all = pd.concat(df_results_all, keys=alpha_list, names=["alpha"])

            df_results_all.to_pickle(f"df_results_all_{rep_name}_{dim}.pkl")

            titles = [r"SNR: $\theta$ gradient ($d =$" + str(dim) + ")", r"SNR: $\phi$ gradient ($d =$" + str(dim) + ")"]
            for i, target_type in enumerate(['_p_grad', '_q_grad']):
                df_results_all[f'vr_iwae{target_type}_snr'].unstack(level=0).plot(logy=True, title=titles[i], xlabel=r"$N$", ylabel="SNR", logx=True, colormap='coolwarm')
                plt.legend(labels=[rf'$\alpha={alpha}$' for alpha in alpha_list], loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(figure_path, f"vr_iwae{target_type}_snr_against_N_{rep_name + '_' if target_type == '_q_grad' else ''}dim_{dim}.png"))
                plt.close()


if __name__ == '__main__':
    main()
