import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import cm
import seaborn as sns
import glob

# Parameters
alpha_list = [0., 0.2, 0.5]
dim_list = [20, 100, 1000]
perturb_sig_list = [0.01, 0.5, 0.]
figure_path = "figures_linear_gaussian"
num_run = 1
datapoints = 1
J = 10

if not os.path.exists(figure_path):
    os.mkdir(figure_path)

# Plot settings
plt.style.use('bmh')
plt.rcParams['figure.facecolor'] = '1'
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams["legend.loc"] = 'lower right'

for perturb_sig in perturb_sig_list:
    for dim in dim_list:
        df_results_all = []
        for alpha in alpha_list:
            df_results_list = []

            # if perturb_sig == 0.:
            #     #print(f"./experiments/linear_gaussian/**/cfg-dim={dim},perturb_sig=0/**/df_results_{alpha}.pkl")
            #     experiment_files_list = sorted(glob.glob(f"./experiments/linear_gaussian/**/cfg-dim={dim},perturb_sig=0/**/df_results_{alpha}.pkl"))
            # else:
            #     #print(f"./experiments/linear_gaussian/**/cfg-dim={dim},perturb_sig={perturb_sig}/**/df_results_{alpha}.pkl")
            #     experiment_files_list = sorted(glob.glob(f"./experiments/linear_gaussian/**/cfg-dim={dim},perturb_sig={perturb_sig}/**/df_results_{alpha}.pkl"))
            experiment_files_list = sorted(glob.glob(f"./experiments/linear_gaussian/2022-09-20/cfg-dim={dim},perturb_sig={perturb_sig}/18-05-18/**/df_results_{alpha}.pkl"))

            assert len(experiment_files_list) == num_run
            for i in range(num_run):
                df_results = pd.read_pickle(experiment_files_list[i])
                df_results.index = 2 ** (df_results.index + 1)
                df_results = df_results.rename_axis("$N$")

                df_results_list.append(df_results)
            df_results_all.append(pd.concat(df_results_list, keys=list(range(num_run)), names=["run"]))
        df_results_all = pd.concat(df_results_all, keys=alpha_list, names=["alpha"])

        # Plot curves
        for alpha in alpha_list:
            for i in range(datapoints):
                B_d = df_results_all.loc[pd.IndexSlice[alpha, 0, 2], f"B_d_datapoints{i}"]
                vr_iwae_means = df_results_all.loc[pd.IndexSlice[alpha, 0, :], f"vr_iwae_mean_datapoints{i}"].to_numpy()
                vr_iwae_1overN = df_results_all.loc[pd.IndexSlice[alpha, 0, :], f"vr_iwae_bound_1overN_datapoints{i}"].to_numpy()

                plt_iters = [2 ** j for j in range(1, J)]

                # 1overN plots
                to_be_added = np.array([1 / j for j in plt_iters])

                if dim == 20:
                    linsp = np.linspace(0, -.2, 100)
                elif dim == 100:
                   linsp = np.linspace(0.5, -0.8, 10)
                else:
                    linsp = np.linspace(0, -0.2, 10)

                cBd1 = iter(cm.Purples(np.linspace(0.15, 0.45, len(linsp))))
                cBd = iter(cm.Greens(np.linspace(0.3, 0.6, len(linsp))))

                plt.figure()
                for lin in linsp:
                    c1 = next(cBd1)
                    add_something_estim = np.add(vr_iwae_1overN, lin * to_be_added)
                    plt.plot(plt_iters, add_something_estim, color=c1)

                plt.plot(plt_iters, vr_iwae_means, label="MC approximation")
                plt.legend(loc='lower right')
                plt.xlabel(r"$N$")
                plt.ylabel(r"$\ell^{(\alpha)}_{N,d}(\theta, \phi; x)$")
                plt.title(r"VR-IWAE bound ($\alpha =$" + str(alpha) + "$, d =$" + str(dim) + ", $\sigma_{\mathrm{perturb}}=$" + str(perturb_sig) + ")")
                plt.tight_layout()
                plt.savefig(figure_path + '/vr_iwae_bound_against_N_1overN_' + str(dim) + "_perturb_sig_" + str(perturb_sig) + "_" + str(alpha) + 'datapoints' + str(i) +'.png')
                plt.close()

        for i in range(datapoints):
            # Approximated Log-normal plots
            to_be_added = np.array([np.sqrt(dim) * np.log(np.log(j)) / np.sqrt(np.log(j)) for j in plt_iters])

            if dim == 20:
                if perturb_sig == 0.5:
                    linsp = np.linspace(-2.65, -2.45, 50)
                elif perturb_sig == 0.01:
                    linsp = np.linspace(-.82, -.8, 50)
                else:
                    linsp = np.linspace(-.82, -.8, 50)

            if dim == 100:
                if perturb_sig == 0.5:
                    linsp = np.linspace(-2.0, -1.75, 50)
                elif perturb_sig == 0.01:
                    linsp = np.linspace(-.65, -.62, 50)
                else:
                    linsp = np.linspace(-.65, -.62, 50)

            if dim == 1000:
                if perturb_sig == 0.5:
                    linsp = np.linspace(-1.35, -1.55, 50)
                elif perturb_sig == 0.01:
                    linsp = np.linspace(-0.38, -0.41, 50)
                else:
                    linsp = np.linspace(-0.38, -0.41, 50)

            cBd = iter(cm.Greens_r(np.linspace(0.5, 0.8, len(linsp))))
            CMapCool = plt.get_cmap('coolwarm')

            cMapCool = iter(CMapCool(np.linspace(0.1, 0.3, len(alpha_list))))

            plt.figure()
            vr_iwae_ApproxLogNormal = df_results_all.loc[
                pd.IndexSlice[0.2, 0, :], f"vr_iwae_bound_approx_LogNormal_datapoints{i}"].to_numpy() - np.log(np.arange(1, J)) / (0.2 - 1)  # The last term was removed since my initial implementation included this term in the saved results

            for lin in linsp:
                c = next(cBd)
                add_something = np.add(vr_iwae_ApproxLogNormal, lin * to_be_added)
                plt.plot(plt_iters[2:], add_something[2:], color=c)

            for alpha in alpha_list:
                cMap = next(cMapCool)
                vr_iwae_means = df_results_all.loc[pd.IndexSlice[alpha, 0, :], f"vr_iwae_mean_datapoints{i}"].to_numpy()

                plt.plot(plt_iters[2:], vr_iwae_means[2:], label= r"$\alpha$ = " + str(alpha), color=cMap)

            plt.legend(loc='lower right', title="MC approximation")
            plt.xlabel(r"$N$")
            plt.ylabel(r"$\ell^{(\alpha)}_{N,d}(\theta, \phi; x)$")
            plt.title(r"VR-IWAE bound ($d =$" + str(dim) + ", $\sigma_{\mathrm{perturb}}=$" + str(perturb_sig) + ")")
            plt.tight_layout()
            plt.savefig(os.path.join(figure_path,
                        'vr_iwae_bound_against_N_ApproxLogNormal_' + str(dim) + "_perturb_sig_" + str(perturb_sig) + "_datapoints" + str(i) + '.png'))
            plt.close()