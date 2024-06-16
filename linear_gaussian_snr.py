import os
from functools import partial
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from statsmodels.graphics.gofplots import qqplot
from sklearn.metrics import mean_squared_error
import hydra
from pathlib import Path

import utils
import losses
from datasets import load_dataset
from log_normal_gaussian import compute_gamma_alpha_LogNormal, compute_gap_1overN, compute_gap_LogNormal


def get_module_grads(module):
    grads = []
    for param in module.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    return grads


def get_log_weights(mu, encoder, x, N):
    pz = Independent(Normal(mu, 1), 1)
    px_z_fn = lambda z: Independent(Normal(z, 1), 1)
    qz_x_fn = lambda x: Independent(Normal(encoder(x), np.sqrt(2 / 3)), 1)

    _, log_ws = losses.vr_iwae_loss_v2(x, pz, px_z_fn, qz_x_fn, N=N)
    return log_ws


def compute_vr_bound_analytic(mu, encoder, data, alpha):
    dim = data.shape[-1]
    if alpha == 1:
        true_vr_bound = - (np.log(3*np.pi)/2 + 1/6) * dim - (encoder(data) - mu).square().sum(1).mean() / 2 - (encoder(data) - data).square().sum(1).mean() / 2
    else:
        true_vr_bound = - (np.log(3*np.pi) + np.log(1 + (1-alpha)/3) / (1-alpha)) * dim / 2 - (encoder(data) - mu).square().sum(1).mean() / 2 - (encoder(data) - data).square().sum(1).mean() / 2 + (1 - alpha) * (2 * encoder(data) - mu - data).square().sum(1).mean() / (4 - alpha) 
    return true_vr_bound


def compute_gamma_alpha_analytic(mu, encoder, x, alpha):
    dim = x.shape[-1]
    if alpha == 1:
        gamma = 0
    else: 
        gamma = 1 / (1 - alpha) * (torch.exp(-(np.log(1 + 2*(1-alpha)/3) / 2 - np.log(1 + (1-alpha)/3)) * dim + ((2 - 2*alpha)**2 / (5 - 2*alpha) - 2*(1 - alpha)**2 / (4 - alpha)) * (2 * encoder(x) - mu - x).square().sum(1)) - 1).mean()
        gamma = gamma.item()
    return gamma


def compute_B_d_analytic(mu, encoder, x):
    true_ll = compute_vr_bound_analytic(mu, encoder, x, 0).item()
    true_elbo = compute_vr_bound_analytic(mu, encoder, x, 1).item()
    mu = true_elbo - true_ll
    B_d = np.sqrt(-2 * mu)
    return B_d


def compute_stdev_log_ws_analytic(mu, encoder, x):
    dim = x.shape[-1]
    var_log_ws = 1/18 * dim + 8/3 * (1/2 *(mu + x) - encoder(x)).square().sum(1)
    return var_log_ws.sqrt().mean().item()


def compute_gap_approx_LogNormal(stdev_log_ws, alpha, B_d, J):
    results_loss = []
    for j in range(1, J):
        results_loss.append(-B_d ** 2 / 2 + stdev_log_ws * np.sqrt(2 * np.log(2 ** j)))
    results_gap = np.array(results_loss)

    return results_gap


def plot_compute_vr_bound(mu, encoder, data):
    dim = data.shape[-1]
    vr_bounds = []
    alphas =  np.linspace(0, 1, 11)
    for alpha in alphas:
        vr_bounds.append(compute_vr_bound_analytic(mu, encoder, data, alpha).item())
    plt.plot(alphas, vr_bounds)

    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\mathcal{L}^{(\alpha)}(\theta, \phi; x)$")
    plt.title(r"True VR bound $\mathcal{L}^{(\alpha)}(\theta, \phi; x)$ ($d=$" + str(dim) + ")")
    plt.tight_layout()
    plt.savefig(f'vr_bound_against_alpha_dim_{dim}.png')

    df_results = pd.DataFrame({'alpha': alphas, 'vr_bound': vr_bounds})
    df_results.to_pickle('df_results_vr_bound.pkl')


@torch.no_grad()
def plot_log_ws(mu, encoder, data, args, dim):
    fig, axes = plt.subplots()
    df_results = pd.DataFrame()
    for i in tqdm(range(args.datapoints)):
        x = data[i:(i+1)]
        log_ws = get_log_weights(mu, encoder, x, args.weight_samples)
        true_ll = Independent(Normal(mu, np.sqrt(2)), 1).log_prob(x)
        log_ws = log_ws - true_ll
        log_ws = log_ws.flatten().detach().numpy()
        df_results[i] = log_ws

        values, bins = np.histogram(log_ws, bins=81, density=True)
        bins = (bins[1:] + bins[:-1]) / 2
        axes.plot(bins, values)

    axes.set_xlabel(r"$\log \overline{w}_i$")
    axes.set_ylabel("Density")
    axes.set_title(r"Distribution of $\log \overline{w}_i$ ($d=$" + str(dim) + ", $\sigma=$" + str(args.perturb_sig) + ")")
    plt.tight_layout()
    plt.savefig(f'weight_distr_{dim}_perturb_sig_{args.perturb_sig}.png')

    df_results.to_pickle('df_results_log_ws.pkl')


def compute_vr_iwae_results(mu, encoder, loss_name, data, pz, px_z_fn, qz_x_fn, qz_x_dist_fn, alpha, args, nb_repeat=10**3):
    optimizer = utils.get_optimizer([mu, *encoder.parameters()], args)

    p_grad_idx = np.rint(np.linspace(0, len(mu)-1, 10)).astype(int)[args.run]
    q_grad_idx = np.rint(np.linspace(0, len(nn.utils.parameters_to_vector(encoder.parameters()))-1, 10)).astype(int)[args.run]

    J = 10

    results_keys = ["true_ll", "true_elbo", "true_vr_bound", "vr_iwae_mean", "vr_iwae_std",
                    "true_ll_p_grad", "true_elbo_p_grad", "true_vr_bound_p_grad", "vr_iwae_p_grad_mean", "vr_iwae_p_grad_std",
                    "true_elbo_q_grad", "true_vr_bound_q_grad", "vr_iwae_q_grad_mean", "vr_iwae_q_grad_std"]
    results = {key: [] for key in results_keys}

    optimizer.zero_grad()
    true_ll = Independent(Normal(mu, np.sqrt(2)), 1).log_prob(data).mean()
    true_ll.backward()
    results["true_ll"] = [true_ll.item()] * (J - 1)
    results["true_ll_p_grad"] = [mu.grad.view(-1)[p_grad_idx].item()] * (J - 1)

    optimizer.zero_grad()
    true_elbo = compute_vr_bound_analytic(mu, encoder, data, 1)
    true_elbo.backward()
    results["true_elbo"] = [true_elbo.item()] * (J - 1)
    results["true_elbo_p_grad"] = [mu.grad.view(-1)[p_grad_idx].item()] * (J - 1)
    results["true_elbo_q_grad"] = [get_module_grads(encoder).view(-1)[q_grad_idx].item()] * (J - 1)

    if alpha == 1:
        results["true_vr_bound"] = [true_elbo.item()] * (J - 1)
        results["true_vr_bound_p_grad"] = [mu.grad.view(-1)[p_grad_idx].item()] * (J - 1)
        results["true_vr_bound_q_grad"] = [get_module_grads(encoder).view(-1)[q_grad_idx].item()] * (J - 1)
    else:
        optimizer.zero_grad()
        true_vr_bound = compute_vr_bound_analytic(mu, encoder, data, alpha)
        true_vr_bound.backward()
        results["true_vr_bound"] = [true_vr_bound.item()] * (J - 1)
        results["true_vr_bound_p_grad"] = [mu.grad.view(-1)[p_grad_idx].item()] * (J - 1)
        results["true_vr_bound_q_grad"] = [get_module_grads(encoder).view(-1)[q_grad_idx].item()] * (J - 1)


    for j in tqdm(range(1, J)):
        results_repeat_keys = ["vr_iwae", "vr_iwae_p_grad", "vr_iwae_q_grad"]
        results_repeat = {key: [] for key in results_repeat_keys}

        for _ in range(nb_repeat):
            optimizer.zero_grad()
            loss = getattr(losses, loss_name)(data, pz, px_z_fn, qz_x_fn, N=2 ** j, alpha=alpha,
                                              encoder=encoder, qz_x_dist_fn=qz_x_dist_fn)[0]
            vr_iwae = -loss
            vr_iwae.backward()

            results_repeat["vr_iwae"].append(vr_iwae.item())
            results_repeat["vr_iwae_p_grad"].append(mu.grad.view(-1)[p_grad_idx].item())
            results_repeat["vr_iwae_q_grad"].append(get_module_grads(encoder).view(-1)[q_grad_idx].item())

        for key in results_repeat_keys:
            results[f"{key}_mean"].append(np.mean(results_repeat[key]))
            results[f"{key}_std"].append(np.std(results_repeat[key]))

    for i in range(args.datapoints):
        x = data[i:(i+1)]
        true_ll = Independent(Normal(mu, np.sqrt(2)), 1).log_prob(x)
        true_vr_bound = compute_vr_bound_analytic(mu, encoder, x, alpha)

        vr_iwae_means = []
        for j in tqdm(range(1, J)):
            vr_iwae_repeat = []
            for _ in range(nb_repeat):
                vr_iwae_repeat.append(- getattr(losses, loss_name)(x, pz, px_z_fn, qz_x_fn, N=2 ** j, alpha=alpha,
                                                                   encoder=encoder, qz_x_dist_fn=qz_x_dist_fn)[0].item())
            vr_iwae_means.append(np.mean(vr_iwae_repeat))
        results[f"vr_iwae_mean_datapoints{i}"] = vr_iwae_means

        gamma_alpha = compute_gamma_alpha_analytic(mu, encoder, x, alpha)
        B_d = compute_B_d_analytic(mu, encoder, x)
        stdev_log_ws = compute_stdev_log_ws_analytic(mu, encoder, x)
        results[f"gamma_alpha_datapoints{i}"] = [gamma_alpha] * (J - 1)
        results[f"B_d_datapoints{i}"] = [B_d] * (J - 1)
        results[f"stdev_log_ws_datapoints{i}"] = [stdev_log_ws] * (J - 1)

        # Comparison with 1/N log behavior predicted by Thm3
        results[f"vr_iwae_bound_1overN_datapoints{i}"] = np.array([true_vr_bound.item() - gamma_alpha / (2 * 2 ** j) for j in range(1, J)])
        # Comparison with 1/N log behavior predicted by Thm3 under log-normal assumption
        #gamma_alpha = compute_gamma_alpha_LogNormal(alpha, B_d)
        #gap_1overN = compute_gap_1overN(gamma_alpha, alpha, B_d, J)
        #results[f"vr_iwae_bound_1overN_LogNormal_datapoints{i}"] = true_ll.item() + gap_1overN
        # Comparison with approximate Log Normal behavior predicted by Thm6 (but this time we need log N/d^{1/3} small)

        # Comparison with Log Normal behavior predicted by Thm5 (but this time we need log N/d^{1/3} small)
        gap_LogNormal = compute_gap_LogNormal(B_d, alpha, J)
        results[f"vr_iwae_bound_LogNormal_datapoints{i}"] = true_ll.item() + gap_LogNormal

        # Comparison with Log Normal behavior predicted by Thm6 (but this time we need log N/d^{1/3} small)
        gap_approx_LogNormal = compute_gap_approx_LogNormal(stdev_log_ws, alpha, B_d, J)
        results[f"vr_iwae_bound_approx_LogNormal_datapoints{i}"] = true_ll.item() + gap_approx_LogNormal

        print(results[f"vr_iwae_bound_LogNormal_datapoints{i}"])
        print(results[f"vr_iwae_bound_approx_LogNormal_datapoints{i}"])

    df_results = pd.DataFrame(results)
    return df_results


@hydra.main(config_path="./conf", config_name="linear_gaussian")
def main(args):
    run = args.run
    dim = args.dim
    seed = args.seed
    N = args.N
    loss_name = args.loss_name
    alpha = args.alpha
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)
    
    if seed is not None:
        utils.manual_seed(seed)

    plt.style.use('bmh')
    plt.rcParams['figure.facecolor'] = '1'
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams["legend.loc"] = 'lower right'

    data = Normal(0, np.sqrt(2)).sample((1024, dim)).to(device)

    true_mu = data.mean(0).to(device)
    true_A = 0.5 * torch.ones(dim).to(device)
    true_b = 0.5 * true_mu.to(device)

    mu = true_mu + args.perturb_sig * torch.randn(dim, device=device)
    mu.requires_grad_()

    class LinearEncoder(nn.Module):
        def __init__(self):
            super().__init__()

            self.A = nn.Parameter(true_A + args.perturb_sig * torch.randn(dim, device=device))
            self.b = nn.Parameter(true_b + args.perturb_sig * torch.randn(dim, device=device))

        def forward(self, x):
            return x * self.A + self.b

    pz = Independent(Normal(mu, 1), 1)
    px_z_fn = lambda z: Independent(Normal(z, 1), 1)
    encoder = LinearEncoder()
    qz_x_fn = lambda x: Independent(Normal(encoder(x), np.sqrt(2 / 3)), 1)
    qz_x_dist_fn = lambda encode: Independent(Normal(encode, np.sqrt(2 / 3)), 1)

    # plot_compute_vr_bound(mu, encoder, data)
    # plot_log_ws(mu, encoder, data, args, dim)

    df_results = compute_vr_iwae_results(mu, encoder, loss_name, data, pz, px_z_fn, qz_x_fn, qz_x_dist_fn, alpha, args)
    df_results.to_pickle("df_results.pkl")


    J = 10

    results_gap = df_results["true_ll"].to_numpy() - df_results["vr_iwae_mean"].to_numpy()
    plt.loglog(2**np.arange(1, J), results_gap)
    plt.loglog(2**np.arange(1, J), results_gap[0] / 2**np.arange(J-1), label="O(1/N)")
    plt.legend()
    plt.savefig("vr_iwae_gap_decay.png")
    plt.close()

    plt.semilogy(df_results["vr_iwae_q_grad_mean"].abs() / df_results["vr_iwae_q_grad_std"])
    plt.title("SNR: q grad")
    plt.savefig("vr_iwae_q_grad_snr.png")

    for i in range(args.datapoints):
        # Log-normal plots
        B_d = df_results[f"B_d_datapoints{i}"].iloc[0]
        vr_iwae_means = df_results[f"vr_iwae_mean_datapoints{i}"].to_numpy()
        vr_iwae_LogNormal = df_results[f"vr_iwae_bound_LogNormal_datapoints{i}"].to_numpy()
        plt_iters = [2 ** j for j in range(1, J)]

        to_be_added = np.array([B_d * np.log(np.log(j)) / np.sqrt(np.log(j)) for j in plt_iters])
        linsp = np.linspace(-2, 0.5, 50)
        cBd = iter(cm.Reds_r(np.linspace(0.6, 0.9, len(linsp))))

        plt.figure()
        for lin in linsp:
            c = next(cBd)
            add_something = np.add(vr_iwae_LogNormal, lin * to_be_added)
            plt.plot(plt_iters[2:], add_something[2:], color=c)

        plt.plot(plt_iters, vr_iwae_means, label="MC approximation")
        plt.legend(loc='lower right')
        plt.xlabel(r"$N$")
        plt.ylabel(r"$\ell^{(\alpha)}_{N,d}(\theta, \phi; x)$")
        plt.title(r"VR-IWAE bound ($\alpha =$" + str(alpha) + "$, d =$" + str(dim) + ")")
        plt.savefig('vr_iwae_bound_against_N_LogNormal_' + str(dim) + "_" + str(
                    alpha) + 'datapoints' + str(i) + '.png')
        plt.close()


if __name__ == '__main__':
    main()