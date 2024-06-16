import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import cm
import seaborn as sns
import glob

# Parameters
alpha_list = [0., 0.2, 0.5]
dim_list = [1, 5, 10, 20, 50, 100, 1000, 2500, 5000, 10000]
figure_path = "./figures_real_data/2022-09-21/cfg-/15-39-17"

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

for dim in dim_list:
    df_results_all = []
    for alpha in alpha_list:
        df_results = pd.read_pickle(f"experiments/mnist/2022-09-21/cfg-/15-39-17/df_vr_iwae_{alpha}_{dim}_0.pkl")[["mean"]]
        df_results["vr_iwae_mean_datapoints0"] = df_results["mean"]
        df_results["vr_iwae_bound_1overN_datapoints0"] = np.load(f"experiments/mnist/2022-09-21/cfg-/15-39-17/vr_iwae_1overN_{alpha}_{dim}_0.npy")
        df_results["vr_iwae_bound_approx_LogNormal_datapoints0"] = np.load(f"experiments/mnist/2022-09-21/cfg-/15-39-17/vr_iwae_approx_LogNormal_{alpha}_{dim}_0.npy")
        df_results_all.append(df_results)
    df_results_all = pd.concat(df_results_all, keys=alpha_list, names=["alpha"])

    # Plot curves
    for alpha in alpha_list:
        for i in range(datapoints):
            vr_iwae_means = df_results_all.loc[pd.IndexSlice[alpha, :], f"vr_iwae_mean_datapoints{i}"].to_numpy()
            vr_iwae_1overN = df_results_all.loc[pd.IndexSlice[alpha, :], f"vr_iwae_bound_1overN_datapoints{i}"].to_numpy()

            plt_iters = [2 ** j for j in range(1, J)]

            # 1overN plots
            to_be_added = np.array([1 / j for j in plt_iters])
            #linsp = np.linspace(4.5, -4.5, 50)

            if dim < 10:
                linsp = np.linspace(-.3, .1, 50)

            elif dim == 10:
                linsp = np.linspace(-1.4, -1., 50)

            elif dim == 20:
                linsp = np.linspace(-1.125, -1.07, 50)

            elif dim == 50:
                linsp = np.linspace(-.75, -0.65, 50)
                if alpha == 0.2:
                    linsp = np.linspace(-.7, -0.6, 50)
                if alpha == 0.:
                    linsp = np.linspace(-.3, -0.2, 50)

            elif dim == 100:
                linsp = np.linspace(-.65, -0.62, 50)
                if alpha == 0.2:
                    linsp = np.linspace(.5, 1.4, 50)

            elif dim == 1000:
                linsp = np.linspace(-0.275, -0.255, 50)

            else:
                linsp = np.linspace(-0.22, -0.19, 50)



            cBd1 = iter(cm.Purples(np.linspace(0.15, 0.45, len(linsp))))
            cBd = iter(cm.Greens_r(np.linspace(0.5, 0.8, len(linsp))))

            plt.figure()
            for lin in linsp:
                c1 = next(cBd1)
                add_something_estim = np.add(vr_iwae_1overN, lin * to_be_added)
                plt.plot(plt_iters, add_something_estim, color=c1)

            plt.plot(plt_iters, vr_iwae_means, label="MC approximation")
            plt.legend(loc='lower right')
            plt.xlabel(r"$N$")
            plt.ylabel(r"$\ell^{(\alpha)}_{N,d}(\theta, \phi; x)$")
            plt.title(r"VR-IWAE bound ($\alpha =$" + str(alpha) + "$, d =$" + str(dim) + ")")
            plt.savefig(figure_path + '/vr_iwae_bound_against_N_1overN_' + str(dim) + "_" + str(alpha) + 'datapoints' + str(i) +'.png')
            plt.close()

    for i in range(datapoints):
        # Approximated Log-normal plots
        to_be_added = np.array([np.sqrt(dim) * np.log(np.log(j)) / np.sqrt(np.log(j)) for j in plt_iters])

        if dim < 10:
            linsp = np.linspace(-2.87, -2.77, 50)

        elif dim == 10:
            linsp = np.linspace(-1.79, -1.7, 50)

        elif dim == 20:
            linsp = np.linspace(-1.125, -1.07, 50)
        
        elif dim == 50:
            linsp = np.linspace(-.88, -0.84, 50)
            
        elif dim == 100:
            linsp = np.linspace(-.67, -0.64, 50)
        
        elif dim == 1000:
            linsp = np.linspace(-0.275, -0.255, 50)

        else:
            linsp = np.linspace(-0.22, -0.19, 50)

        cBd = iter(cm.Greens_r(np.linspace(0.5, 0.8, len(linsp))))
        CMapCool = plt.get_cmap('coolwarm')

        cMapCool = iter(CMapCool(np.linspace(0.1, 0.3, len(alpha_list))))

        plt.figure()
        vr_iwae_ApproxLogNormal = df_results_all.loc[
            pd.IndexSlice[0.2, :], f"vr_iwae_bound_approx_LogNormal_datapoints{i}"].to_numpy()

        for lin in linsp:
            c = next(cBd)
            add_something = np.add(vr_iwae_ApproxLogNormal, lin * to_be_added)
            plt.plot(plt_iters[2:], add_something[2:], color=c)

        for alpha in alpha_list:
            cMap = next(cMapCool)
            vr_iwae_means = df_results_all.loc[pd.IndexSlice[alpha, :], f"vr_iwae_mean_datapoints{i}"].to_numpy()

            plt.plot(plt_iters[1:], vr_iwae_means[1:], label= r"$\alpha$ = " + str(alpha), color=cMap)

        plt.legend(loc='lower right', title="MC approximation")
        plt.xlabel(r"$N$")
        plt.ylabel(r"$\ell^{(\alpha)}_{N,d}(\theta, \phi; x)$")
        plt.title(r"VR-IWAE bound ($d =$" + str(dim) + ")")
        plt.tight_layout()
        plt.savefig(os.path.join(figure_path,
                    'vr_iwae_bound_against_N_ApproxLogNormal_' + str(dim) + "_datapoints" + str(i) + '.png'))
        plt.close()