import os
import re
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot

plt.style.use('bmh')
plt.rcParams['figure.facecolor'] = '1'
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams["legend.loc"] = 'lower right'

datapoints = 1

dim_list = [20, 100, 1000]
pertub_sig_list = [0.5, 0.01, 0.0]
overlapp = 0.6
experiment_file_dir = "figures_linear_gaussian"

for dim in dim_list:
    for i in range(datapoints):
        fig, axes = plt.subplots()
        for perturb_sig in pertub_sig_list:
            
            experiment_files_list = sorted(glob.glob(f"./experiments/linear_gaussian/2022-09-20/cfg-dim={dim},perturb_sig={perturb_sig}/18-05-18/**/df_results_log_ws.pkl"))

            for experiment_file in experiment_files_list:
                df_results = pd.read_pickle(experiment_file)

                log_ws = df_results[i]
                values, bins = np.histogram(log_ws, bins=81, density=True)
                bins = (bins[1:] + bins[:-1]) / 2

                if perturb_sig == 0.:
                    axes.plot(bins, values, label=r"$\sigma_{\mathrm{perturb}}$ =" + str(perturb_sig))
                else:
                    axes.plot(bins, values, label=r"$\sigma_{\mathrm{perturb}}$ =" + str(perturb_sig))

        axes.set_xlabel(r"$\log \overline{w}_i$")
        axes.set_ylabel("Density")
        axes.set_title(r"Distribution of $\log \overline{w}_i$ ($d=$" + str(dim) + ")")
        plt.tight_layout()
        plt.legend(loc='upper left')
        plt.savefig(experiment_file_dir + '/weight_distr_' + str(dim) + '_'+ str(i) + '.png')
        plt.close()

        fig, axes = plt.subplots()
        for j, perturb_sig in enumerate(pertub_sig_list):

            experiment_files_list = sorted(glob.glob(f"./experiments/linear_gaussian/2022-09-20/cfg-dim={dim},perturb_sig={perturb_sig}/18-05-18/**/df_results_log_ws.pkl"))
            
            for experiment_file in experiment_files_list:
                df_results = pd.read_pickle(experiment_file)

                log_ws = df_results[i]
                qqplot(log_ws, line='45', fit=True, ax=axes, markerfacecolor=f"C{j}", markeredgecolor=f"C{j}", markersize=5, label=r"$\sigma_{\mathrm{perturb}}$ = " + str(perturb_sig))
        plt.legend(loc="upper left")
        plt.title(r"QQ-plot ($d=$" + str(dim) + ")")
        plt.savefig(experiment_file_dir + '/qqplot_' + str(dim) + "_" + str(i) + '.png')
        plt.close()
