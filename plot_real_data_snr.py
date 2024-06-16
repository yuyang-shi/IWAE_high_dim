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


def main():
    # Plot settings
    plt.style.use('bmh')
    plt.rcParams['figure.facecolor'] = '1'
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams["legend.loc"] = 'lower right'

    # Make subdirectory of ./figure/ for run
    # figure_path = 'figures_real_data/2022-09-22/cfg-/00-34-50/0'
    # experiment_path = 'experiments/mnist/2022-09-22/cfg-/00-34-50/0'
    figure_path = 'figures_real_data/2022-09-22/cfg-/13-17-59/0'
    experiment_path = 'experiments/mnist/2022-09-22/cfg-/13-17-59/0'

    try:
        os.makedirs(figure_path)
    except OSError:
        pass

    # Load model and data
    dim_list = [1, 5, 10, 20, 50, 100, 1000, 2500, 5000, 10000]
    alpha_list = [0, 0.2, 0.5, 0.8]
    loss_name_list = ["vr_iwae_dreg_loss_v6"]  # ["vr_iwae_loss_v2"]  # 
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
            df_results_all = pd.read_pickle(os.path.join(experiment_path, f"df_results_all_{rep_name}_{dim}.pkl")).loc[alpha_list]

            titles = [r"SNR: $\theta$ gradient ($d =$" + str(dim) + ")", r"SNR: $\phi$ gradient ($d =$" + str(dim) + ")"]
            for i, target_type in enumerate(['_p_grad', '_q_grad']):
                df_results_all[f'vr_iwae{target_type}_snr'].unstack(level=0).plot(logy=True, title=titles[i], xlabel=r"$N$", ylabel="SNR", logx=True, colormap='coolwarm')
                if target_type == '_p_grad':
                    plt.loglog(2**np.arange(1, J), df_results_all[f'vr_iwae{target_type}_snr'].groupby(level=1).mean().loc[pd.IndexSlice[2]] * 2**(np.arange(J-1)/2), ls='-.', c='k')
                    plt.legend(labels=[rf'$\alpha={alpha}$' for alpha in alpha_list] + ["$\Theta(\sqrt{N})$"], loc='upper left')
                elif loss_name == "vr_iwae_dreg_loss_v6":
                    plt.legend(labels=[rf'$\alpha={alpha}$' for alpha in alpha_list], loc='upper left')
                    # plt.legend(labels=[rf'$\alpha={alpha}$' for alpha in alpha_list] + ["$\Theta(\sqrt{N})$"], loc='upper left')
                else:
                    plt.loglog(2**np.arange(1, J), df_results_all[f'vr_iwae{target_type}_snr'].groupby(level=1).mean().loc[pd.IndexSlice[2]] * 2**(np.arange(J-1)/2), ls='-.', c='k')
                    plt.loglog(2**np.arange(1, J), df_results_all[f'vr_iwae{target_type}_snr'].groupby(level=1).mean().loc[pd.IndexSlice[2]] * 2**(-np.arange(J-1)/2), ls=':', c='k')
                    plt.legend(labels=[rf'$\alpha={alpha}$' for alpha in alpha_list] + ["$\Theta(\sqrt{N})$", "$\Theta(1/\sqrt{N})$"], loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(figure_path, f"vr_iwae{target_type}_snr_against_N_{rep_name + '_' if target_type == '_q_grad' else ''}dim_{dim}.png"))
                plt.close()


if __name__ == '__main__':
    main()
