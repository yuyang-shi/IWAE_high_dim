import os
from functools import partial
import hydra
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm
import torch
import pandas as pd

import losses
from datasets import load_dataset
from utils import logmeanexp, lognormexp, manual_seed
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
    alpha_list = [0., 0.2, 0.5]
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(device)

    J = 10
    plt_iters = [2 ** j for j in range(1, J)]
    nb_runs = 100
    # nb_samples_baseline = 2000
    epoch = 0

    for dim in dim_list:
        for alpha in alpha_list:
            vae, test_loader = get_model_for_plotting(args, dim, alpha)
            vae = vae.to(device)
            test_loader_iter = iter(test_loader)

            for i in range(args.datapoints):
                x, _ = next(test_loader_iter)
                x = x.to(device)

                mu, stdev_log_ws_estim, gamma_alpha_estim, baseline_VR_bound_estim, baseline_log_lik_estim = log_ws_plots_and_computations(vae, x, args, figure_path, dim, i, alpha)
                B_d_estim = np.sqrt(-2 * mu)

                vr_iwae = compute_vr_iwae_bound(vae, args, J, x, alpha)
                df_vr_iwae = pd.DataFrame({'Run 0': vr_iwae})

                for nb in range(1, nb_runs):
                    run_nb = 'Run ' + str(nb + 1)
                    vr_iwae = compute_vr_iwae_bound(vae, args, J, x, alpha)
                    df_vr_iwae[run_nb] = vr_iwae

                df_vr_iwae['mean'] = df_vr_iwae.mean(axis=1)

                df_vr_iwae.to_pickle(f"df_vr_iwae_{alpha}_{dim}_{epoch}.pkl")

                # baseline_VR_bound_estim = baseline_VR_bound(x, vae, nb_samples=nb_samples_baseline, alpha=alpha, nb_repeat=nb_runs)
                # baseline_log_lik_estim = baseline_VR_bound(x, vae, nb_samples=nb_samples_baseline, alpha=0.0, nb_repeat=nb_runs)

                # Comparison with 1/N log behavior predicted by Thm3
                vr_iwae_estim_1overN_estim = np.array([baseline_VR_bound_estim - gamma_alpha_estim / (2 * 2 ** j) for j in range(1, J)])
                filename = "vr_iwae_1overN_" + str(alpha) + "_" + str(dim) + "_" + str(epoch) + ".npy"
                np.save(filename, vr_iwae_estim_1overN_estim)

                # Comparison with 1/N log behavior predicted by Thm3 under log-normal assumption
                gamma_alpha = compute_gamma_alpha_LogNormal(alpha, B_d_estim)
                gap_1overN = compute_gap_1overN(gamma_alpha, alpha, B_d_estim, J)
                vr_iwae_estim_1overN = baseline_log_lik_estim + gap_1overN
                filename = "vr_iwae_1overN_under_LogNormal_" + str(alpha) + "_" + str(dim) + "_"  + str(epoch) + ".npy"
                np.save(filename, vr_iwae_estim_1overN)

                # Comparison with Log Normal behavior predicted by Thm6 (but this time we need log N/d^{1/3} small)
                gap_approx_LogNormal = compute_gap_approx_LogNormal(stdev_log_ws_estim, alpha, B_d_estim, J)
                vr_iwae_approx_LogNormal = baseline_log_lik_estim + gap_approx_LogNormal
                filename = "vr_iwae_approx_LogNormal_" + str(alpha) + "_" + str(dim) + "_" + str(epoch) + ".npy"
                np.save(filename, vr_iwae_approx_LogNormal)

                # Plotting

                # 1overN plots
                to_be_added = np.array([1 / j for j in plt_iters])
                linsp = np.linspace(4.5, -4.5, 50)
                cBd1 = iter(cm.Purples(np.linspace(0.3, 0.6, len(linsp))))

                plt.figure()
                for lin in linsp:
                    c1 = next(cBd1)
                    add_something_estim = np.add(vr_iwae_estim_1overN_estim, lin * to_be_added)
                    plt.plot(plt_iters, add_something_estim, color=c1)

                plt.plot(plt_iters, df_vr_iwae['mean'], label="MC approximation")
                plt.legend(loc='lower right')
                plt.xlabel(r"$N$")
                plt.ylabel(r"$\ell^{(\alpha)}_{N,d}(\theta, \phi; x)$")
                plt.title(r"VR-IWAE bound ($\alpha =$" + str(alpha) + "$, d =$" + str(dim) + ")")
                plt.savefig(figure_path + '/vr_iwae_bound_against_N_1overN_' + str(dim) + "_" + str(alpha) + 'datapoints' + str(i) +'.png')
                plt.close()

                # Log-normal plots
                to_be_added = np.array([stdev_log_ws_estim * np.log(np.log(j)) / np.sqrt(np.log(j)) for j in plt_iters])
                linsp = np.linspace(-2.5, 0, 50)
                cBd = iter(cm.Greens(np.linspace(0.3, 0.6, len(linsp))))

                plt.figure()
                for lin in linsp:
                    c = next(cBd)
                    add_something = np.add(vr_iwae_approx_LogNormal, lin * to_be_added)
                    plt.plot(plt_iters[2:], add_something[2:], color=c)

                plt.plot(plt_iters, df_vr_iwae['mean'], label="MC approximation")
                plt.legend(loc='lower right')
                plt.xlabel(r"$N$")
                plt.ylabel(r"$\ell^{(\alpha)}_{N,d}(\theta, \phi; x)$")
                plt.title(r"VR-IWAE bound ($\alpha =$" + str(alpha) + "$, d =$" + str(dim) + ")")
                plt.savefig(
                    figure_path + '/vr_iwae_bound_against_N_approx_LogNormal_' + str(dim) + "_" + str(
                        alpha) + 'datapoints' + str(
                        i) + '.png')
                plt.close()


if __name__ == '__main__':
    main()
