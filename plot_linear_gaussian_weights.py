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
experiment_files_list = glob.glob("experiments/linear_gaussian/2022-09-08/*/*/df_results_log_ws.pkl")

for experiment_file in experiment_files_list:
    experiment_file_dir = os.path.dirname(experiment_file)
    df_results = pd.read_pickle(experiment_file)
    dim = int(re.search(r"dim=\d+", experiment_file_dir).group()[4:])
    perturb_sig = re.search(r"perturb_sig=\d*\.?\d*", experiment_file_dir)
    perturb_sig = float(perturb_sig.group()[12:]) if perturb_sig is not None else 0.01
    
    print(df_results.shape)

    fig, ax = plt.subplots()
    for i in range(min(datapoints, df_results.shape[1])):
        log_ws = df_results[i]
        qqplot(log_ws, line='45', fit=True, ax=ax, markerfacecolor=f"C{i}", markeredgecolor=f"C{i}")
    plt.title(r"QQ-plot ($d=$" + str(dim) + ", $\sigma_{\mathrm{perturb}}=$" + str(perturb_sig) + ")")
    plt.savefig(experiment_file_dir + '/qqplot_'+ str(dim) + "_perturb_sig_" + str(perturb_sig) + '.png')
    plt.close()

    fig, ax = plt.subplots()
    for i in range(min(datapoints, df_results.shape[1])):
        log_ws = df_results[i]

        values, bins = np.histogram(log_ws, bins=81, density=True)
        bins = (bins[1:] + bins[:-1]) / 2
        ax.plot(bins, values)

    ax.set_xlabel(r"$\log \overline{w}_i$")
    ax.set_ylabel("Density")
    ax.set_title(r"Distribution of $\log \overline{w}_i$ ($d=$" + str(dim) + ", $\sigma_{\mathrm{perturb}}=$" + str(perturb_sig) + ")")
    plt.tight_layout()
    plt.savefig(experiment_file_dir + f'/weight_distr_{dim}_perturb_sig_{perturb_sig}.png')
