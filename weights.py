import os
import torch
from functools import partial
import pytorch_lightning as pl
import hydra
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm
import torch
import pandas as pd
import glob

import losses
from datasets import load_dataset
from utils import logmeanexp, lognormexp, manual_seed
from models import MLPVAE
from log_normal_gaussian import compute_gamma_alpha_LogNormal, compute_gap_1overN, compute_gap_LogNormal
from linear_gaussian_snr import compute_gap_approx_LogNormal


def get_model_for_plotting(args, dim=None):
    # Reads config and returns a model for plotting, plus test_loader
    # If checkpoint is passed, uses checkpointed model
    # Else, creates a new untrained model from scratch

    seed = args.seed
    N = args.N
    loss_name = args.loss_name
    alpha = args.alpha

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
def log_ws_plots_and_computations(model, x, args, figure_path, dim, i, epoch):
    # Plots the distribution of the log weights
    # Returns estimated mean and std of associated normal distribution
    # as well as an estimation of gamma_alpha

    fig, axes = plt.subplots()
    log_ws = get_log_weights(model, x, args.weight_samples)
    logmeanexp_log_ws = logmeanexp(log_ws, dim=0)

    alpha = args.alpha
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
    plt.title(r"Distribution of $\log \overline{w}_i$ ($d =$" + str(dim) + ", epoch=" + str(epoch) + ")")
    plt.savefig(figure_path + '/weight_distr_'+ str(dim) + 'datapoints' + str(i) + "_" + str(epoch) +'.png')
    plt.close()

    qqplot(log_ws, line='45', fit=True)
    plt.title(r"QQ-plot ($d=$" + str(dim) + ", epoch=" + str(epoch) + ")")
    plt.savefig(figure_path + '/qqplot_'+ str(dim) + 'datapoints' + str(i) + "_" + str(epoch) +'.png')
    plt.close()

    mu, std = norm.fit(log_ws)

    return mu, std, gamma.detach().cpu().numpy(), baseline_VR_bound_estim, baseline_log_lik_estim


@torch.no_grad()
def compute_vr_iwae_bound(model, args, J, x):
    alpha = args.alpha

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


@torch.no_grad()
def plot_vr_iwae_bound_single_model(model, test_loader, args, figure_path):
    # Plots the VR-IWAE bound against N, the number of weight samples used

    Ns = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    alpha = args.alpha
    fig, axes = plt.subplots()

    for _ in range(args.datapoints):

        x, _ = next(iter(test_loader))
        x = x.view(x.size(0), *model.obs_dim)[0].unsqueeze(dim=0)

        baselines = []
        for _ in tqdm(range(100)):
            loss, _ = losses.vr_iwae_loss_v2(x, model.pz, model.px_z_fn, model.qz_x_fn, N=10000, alpha=alpha)
            baselines.append(loss.detach().numpy())
        baseline = sum(baselines)/len(baselines)
        
        x = x.repeat(1000,1)
        N_plot = []
        gaps = []
        for N in Ns:
            for _ in tqdm(range(10)):
                loss, _ = losses.vr_iwae_loss_v2(x, model.pz, model.px_z_fn, model.qz_x_fn, N=N, alpha=alpha)
                gap = -loss.detach().numpy() + baseline
                gaps.append(gap)
                N_plot.append(N)

        axes.scatter(N_plot, gaps)

    axes.set_xlabel(r"$N$")
    axes.set_ylabel(r"$\Delta^{(\alpha)}_N$")
    plt.savefig(figure_path + '/vr_iwae_bound_against_N.png')


def plot_vr_iwae_bound(args, figure_path, vary_dims=False):
    # Plots the VR-IWAE bound against N, the number of weight samples used, where we
    # regenerate the model at each step and the latent dimesion is allowed to vary

    Ns = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    if vary_dims:
        dims = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    else:
        dims = [500 for N in Ns]

    N_plot = []
    alpha = args.alpha
    fig, axes = plt.subplots()

    # Get dataset
    _, _, test_loader = load_dataset(args)

    for _ in range(args.datapoints):
        # Get datapoint
        x, _ = next(iter(test_loader))
        x = x.view(x.size(0), *args.obs_dim)[0].unsqueeze(dim=0)

        gaps = []

        for N, dim in zip(Ns, dims):
            # Iterate over number of weight samples
            for _ in range(10):
                # Average over 10 different models
                args.latent_dim = dim
                model, _ = get_model_for_plotting(args)
                print("N: %s, dim: %s" % (N, dim))

                # Estimate true log likelihood baseline
                baselines = []
                for _ in range(100):
                    loss, _ = losses.vr_iwae_loss_v2(x, model.pz, model.px_z_fn, model.qz_x_fn, N=10000, alpha=0.0)
                    baselines.append(loss.detach().numpy())
                baseline = sum(baselines)/len(baselines)

                # Debugging output
                # baselines = np.array(baselines)
                # baselines.sort()
                # print(baselines)
                # print(baseline)

                # Calculate average variational gap over 1000 samples of N weights each
                # Repeat 3 times to get a sense of variance of the measurement of the gap
                x_repeated = x.repeat(1000,1)
                for _ in range(3):
                    loss, _ = losses.vr_iwae_loss_v2(x_repeated, model.pz, model.px_z_fn, model.qz_x_fn, N=N, alpha=alpha)
                    gap = -loss.detach().numpy() + baseline
                    gaps.append(gap)
                    N_plot.append(N)
                    print(gap)

        axes.scatter(N_plot, gaps)

    axes.set_xlabel(r"$N$")
    axes.set_ylabel(r"$\Delta^{(\alpha)}_N$")
    plt.savefig(figure_path + '/vr_iwae_bound_against_N.png')


# def latent_dim(args, figure_path):
# 
#     # Set up data
#     _, _, test_loader = load_dataset(args)
#     x, _ = next(iter(test_loader))
# 
#     dims = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
#     stds = []
# 
#     for dim in dims:
#         model, _ = get_model_for_plotting(args, dim=dim)
#         stds.append(stdev_log_ws(model, x, args.weight_samples))
# 
#     fig, axes = plt.subplots()
#     axes.scatter(dims, stds)
#     axes.set_xlabel("Latent Dimension")
#     axes.set_ylabel(r"Variance of $\log w_i$")
#     plt.savefig(figure_path + '/latent_dim_dependence.png')


@hydra.main(config_path="./conf", config_name="weights")
def main(args):
    # Plot settings
    plt.style.use('bmh')
    plt.rcParams['figure.facecolor'] = '1'
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams["legend.loc"] = 'lower right'

    # Make subdirectory of ./figure/ for run
    figure_dir = hydra.utils.to_absolute_path('figures')
    run_name = os.path.normpath(os.path.relpath(os.getcwd(), os.path.join(
            hydra.utils.to_absolute_path(args.paths.experiments_dir_name), args.name))).replace("\\", "/")
    figure_path = figure_dir + '/' + run_name

    try:
        os.makedirs(figure_path)
    except OSError:
        pass

    if args.checkpoint_path is not None:
        checkpoint_paths = [hydra.utils.to_absolute_path(args.checkpoint_path)]
    else:
        checkpoint_paths = glob.glob(os.path.join(hydra.utils.to_absolute_path(args.checkpoint_dir), "epoch=*.ckpt"))
    epochs = [int(os.path.basename(checkpoint_path).split("-")[0][6:])+1 for checkpoint_path in checkpoint_paths]
    epochs, checkpoint_paths = zip(*sorted(zip(epochs, checkpoint_paths)))
    print(epochs, checkpoint_paths)

    for epoch, checkpoint_path in zip(epochs, checkpoint_paths):
        # Load model and data
        alpha = args.alpha
        dim = args.latent_dim
        seed = args.seed
        args.checkpoint_path = checkpoint_path
        vae, test_loader = get_model_for_plotting(args)

        if seed is not None:
            manual_seed(seed)

        J = 10
        plt_iters = [2 ** j for j in range(1, J)]
        nb_runs = 100
        # nb_samples_baseline = 5000

        test_loader_iter = iter(test_loader)

        for i in range(args.datapoints):
            x, _ = next(test_loader_iter)
            mu, stdev_log_ws_estim, gamma_alpha_estim, baseline_VR_bound_estim, baseline_log_lik_estim = log_ws_plots_and_computations(vae, x, args, figure_path, dim, i, epoch)
            B_d_estim = np.sqrt(-2 * mu)

            print(gamma_alpha_estim)

            vr_iwae = compute_vr_iwae_bound(vae, args, J, x)
            df_vr_iwae = pd.DataFrame({'Run 0': vr_iwae}, index=2**np.arange(1, J))

            for nb in range(1, nb_runs):
                run_nb = 'Run ' + str(nb + 1)
                vr_iwae = compute_vr_iwae_bound(vae, args, J, x)
                df_vr_iwae[run_nb] = vr_iwae

            df_vr_iwae['mean'] = df_vr_iwae.mean(axis=1)

            # Plot variational gap vs N
            vr_iwae_gap = baseline_log_lik_estim - df_vr_iwae['mean']
            vr_iwae_gap.plot(title=r"Variational gap ($\alpha=$" + str(alpha) + ", $d=$" + str(dim) + ", $N=$" + str(args.N) + ")", xlabel=r"$N$", ylabel=r"$\Delta^{(\alpha)}_{N,d}(\theta, \phi; x)$", logx=True, logy=True)
            plt.loglog(2**np.arange(1, J), vr_iwae_gap.iloc[0] / 2**np.arange(J-1), ls=':', c='k')
            plt.legend(labels=[rf'$\alpha={alpha}$', "$O(1/N)$"], loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(figure_path, f"vr_iwae_gap_against_N_dim_" + str(dim) + "_" + str(alpha) + 'datapoints' + str(i) + "_" + str(epoch) + ".png"))
            plt.close()
            plt.autoscale(True)

            # baseline_VR_bound_estim = baseline_VR_bound(x, vae, nb_samples=nb_samples_baseline, alpha=alpha, nb_repeat=nb_runs)
            # baseline_log_lik_estim = baseline_VR_bound(x, vae, nb_samples=nb_samples_baseline, alpha=0.0, nb_repeat=nb_runs)

            # Comparison with 1/N log behavior predicted by Thm3
            vr_iwae_estim_1overN_estim = np.array([baseline_VR_bound_estim - gamma_alpha_estim / (2 * 2 ** j) for j in range(1, J)])

            # Comparison with 1/N log behavior predicted by Thm3 under log-normal assumption
            gamma_alpha = compute_gamma_alpha_LogNormal(alpha, B_d_estim)
            gap_1overN = compute_gap_1overN(gamma_alpha, alpha, B_d_estim, J)
            vr_iwae_estim_1overN = baseline_log_lik_estim + gap_1overN

            # Comparison with Log Normal behavior predicted by Thm6 (but this time we need log N/d^{1/3} small)
            gap_approx_LogNormal = compute_gap_approx_LogNormal(stdev_log_ws_estim, alpha, B_d_estim, J)
            vr_iwae_approx_LogNormal = baseline_log_lik_estim + gap_approx_LogNormal

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
            plt.title(r"VR-IWAE bound ($\alpha =$" + str(alpha) + "$, d =$" + str(dim) + ", epoch=" + str(epoch) + ")")
            plt.savefig(figure_path + '/vr_iwae_bound_against_N_1overN_' + str(dim) + "_" + str(alpha) + 'datapoints' + str(i) + "_" + str(epoch) +'.png')
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
            plt.title(r"VR-IWAE bound ($\alpha =$" + str(alpha) + "$, d =$" + str(dim) + ", epoch=" + str(epoch) + ")")
            plt.savefig(
                figure_path + '/vr_iwae_bound_against_N_approx_LogNormal_' + str(dim) + "_" + str(
                    alpha) + 'datapoints' + str(
                    i) + "_" + str(epoch) + '.png')
            plt.close()

        # Make plots
        # plot_vr_iwae_bound_single_model(vae, test_loader, args, figure_path)
        # plot_gamma_alpha(vae, test_loader, args, figure_path)
        # plot_vr_bound(vae, test_loader, args, figure_path)
        # plot_log_ws(vae, test_loader, args, figure_path)
        # latent_dim(args, figure_path)
        # plot_vr_iwae_bound(args, figure_path, vary_dims=True)`


if __name__ == '__main__':
    main()
