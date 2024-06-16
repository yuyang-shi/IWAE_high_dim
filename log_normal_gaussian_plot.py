import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from matplotlib.pyplot import cm
from pathlib import Path

# Plot settings
plt.style.use('bmh')
plt.rcParams['figure.facecolor'] = '1'
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams["legend.loc"] = 'lower right'


def aux_plot_gap(params_folder, str_gap="gap_", epoch=0):
    experiments_folder = './experiments/log_normal_gaussian/2022-09-29/' + params_folder + '/22-32-02/0/' + str_gap + str(epoch) + '.npy'
    files_retrieved = glob.glob(experiments_folder)
    nb_run = len(files_retrieved)

    # Put the file retrieved inside a df
    filename = files_retrieved[0]
    file_content = np.load(filename)
    df = pd.DataFrame({'Run 1': file_content})

    for i in range(nb_run):
        filename = files_retrieved[i]
        file_content = np.load(filename)

        run_nb = 'Run ' + str(i + 1)
        df[run_nb] = file_content.tolist()

    # Compute mean and quantiles
    df['mean'] = df.mean(axis=1)
    df['q10'] = df.quantile(q=0.10, axis=1)
    df['q90'] = df.quantile(q=0.90, axis=1)

    return df['mean'], df['q10'], df['q90']


def aux_plot_mse_mu_phi(params_folder, str_mu_phi="mse_mu_phi"):
    experiments_folder = './experiments/log_normal_gaussian/2022-09-29/' + params_folder + '/22-32-02/0/' + str_mu_phi + '.npy'
    files_retrieved = glob.glob(experiments_folder)
    nb_run = len(files_retrieved)

    # Put the file retrieved inside a df
    filename = files_retrieved[0]
    file_content = np.load(filename)
    df = pd.DataFrame({'Run 1': file_content})

    for i in range(nb_run):
        filename = files_retrieved[i]
        file_content = np.load(filename)

        run_nb = 'Run ' + str(i + 1)
        df[run_nb] = file_content.tolist()

    # Compute mean and quantiles
    df['mean'] = df.mean(axis=1)
    df['q10'] = df.quantile(q=0.10, axis=1)
    df['q90'] = df.quantile(q=0.90, axis=1)

    return df['mean'], df['q10'], df['q90']



if __name__ == '__main__':

    # Parameters
    alpha_list = [0.2]
    N = 100
    J = 10

    dim_list = [10, 100, 1000]
    loss_list = [1, 2]
    max_epochs = 6000

    for dim in dim_list:
        for loss in loss_list:
            if loss == 1:
                loss_name_str = "vr_iwae_rep"
            else:
                loss == "vr_iwae_drep"


            fig_folder = "./figures_log_normal_gaussian/" + str(dim) + "/loss" + str(loss) + "/"
            Path(fig_folder).mkdir(parents=True, exist_ok=True)

            ## Plot B_d^2
            mse_list = []
            cBd = iter(cm.Blues_r(np.linspace(0.3, 0.7, len(alpha_list))))
            plt.figure()
            for alpha in alpha_list:
                c = next(cBd)

                if alpha == 0.:
                    str_alpha = "0.0"
                else:
                    str_alpha = str(alpha)
                params_folder = 'cfg-N=' + str(N) + ',alpha=' + str_alpha + ',dim=' + str(dim) \
                                + ',loss=' + str(loss) + ',max_epochs=' + str(max_epochs)

                freq = 100
                plt_iters = [0] + [1] + [i for i in range(1, max_epochs + 1) if i % freq == 0]
                mse_, mse_10, mse_90 = aux_plot_mse_mu_phi(params_folder)
                mse_list.append(mse_.to_numpy())

                plt.plot(plt_iters, mse_, label=r"$\alpha =$" + str(alpha), color=c)
                plt.fill_between(plt_iters, mse_10, mse_90, alpha=0.2, color=c)
            plt.xlabel("iterations")
            plt.ylabel(r"$B_d^2/d$")
            plt.title(r'$B_d^2/d$ during training ($d =$' + str(dim) + r", $N=$" + str(N) + ")")
            plt.legend(loc='upper right')
            plt.savefig(fig_folder + "fig_Bd" + str(N) + "_" + "_" + str(dim) + "_" + str(loss) + "_" + str(max_epochs) + ".png")
            plt.close()


            ## Plot gap
            freq_target = 500
            freq = 100
            epochs = [0] + [1] + [i for i in range(1, max_epochs + 1) if i % freq_target == 0]
            epochs_converted_to_get_Bd = [0] + [1] + [int(i/freq) for i in range(1, max_epochs + 1) if i % freq_target == 0]

            ## Plot gap with log normal behavior
            for i, alpha in enumerate(alpha_list):
                mse_alpha = mse_list[i]

                if alpha == 0.:
                    str_alpha = "0.0"
                else:
                    str_alpha = str(alpha)
                params_folder = 'cfg-N=' + str(N) + ',alpha=' + str_alpha + ',dim=' + str(dim) \
                                + ',loss=' + str(loss) + ',max_epochs=' + str(max_epochs)

                for j, epoch in enumerate(epochs):
                    str_plot = r'$\Delta_{N,d}^{(\alpha)}(\theta, \phi; x)$'
                    str_plot_name = r'Variational gap $(\alpha =$' + str(alpha) + r'$, d=$' + str(dim) + ", epoch=" + str(epoch) + ')'
                    gap_, gap_10, gap_90 = aux_plot_gap(params_folder, "gap_MC_", epoch)
                    gap_thm6_, gap_thm6_10, gap_thm_6_90 = aux_plot_gap(params_folder, "gap_LogNormal_", epoch)

                    plt_iters = [2 ** j for j in range(1, J)]

                    plt.figure()

                    B_d = np.sqrt(dim * mse_alpha[epochs_converted_to_get_Bd[j]])
                    to_be_added = np.array([B_d * np.log(np.log(j))/np.sqrt(np.log(j)) for j in plt_iters])

                    if dim == 10:
                        linsp = np.linspace(-2.,-1.5, 50)
                    else:
                        linsp = np.linspace(-1.5,-.5, 50)

                    cBd = iter(cm.Reds_r(np.linspace(0.6, 0.9, len(linsp))))

                    for lin in linsp:
                        c = next(cBd)
                        add_something = np.add(gap_thm6_.to_numpy(), lin * to_be_added)
                        plt.plot(plt_iters[2:], add_something[2:], color=c)

                    plt.plot(plt_iters[2:], gap_[2:], label="MC approximation")

                    plt.xlabel(r"$N$")
                    plt.ylabel(str_plot)
                    plt.legend(loc='lower right')
                    plt.title(str_plot_name)
                    plt.savefig(fig_folder + 'fig_gap_comparison_LogNormal'+ str(N) + "_" + str(alpha) + "_" + str(dim) + "_" + str(loss) + "_" + str(max_epochs) + "_" + str(epoch) +".png")
                    plt.close()

            # Plot gap with 1 over N behavior
            i = 0
            for alpha in alpha_list:
                mse_alpha = mse_list[i]

                if alpha == 0.:
                    str_alpha = "0.0"
                else:
                    str_alpha = str(alpha)
                params_folder = 'cfg-N=' + str(N) + ',alpha=' + str_alpha + ',dim=' + str(dim) \
                                + ',loss=' + str(loss) + ',max_epochs=' + str(max_epochs)

                for epoch in epochs:
                    str_plot = r'$\Delta_{N,d}^{(\alpha)}(\theta, \phi; x)$'
                    str_plot_name = r'Variational gap $(\alpha =$' + str(alpha) + r'$, d=$' + str(dim) + ", epoch=" + str(
                        epoch) + ')'
                    gap_, gap_10, gap_90 = aux_plot_gap(params_folder, "gap_MC_", epoch)
                    gap_thm3_, gap_thm3_10, gap_thm_3_90 = aux_plot_gap(params_folder, "gap_1overN_", epoch)

                    plt_iters = [2 ** j for j in range(1, J)]

                    plt.figure()

                    to_be_added = np.array([1/ j for j in plt_iters])
                    linsp = np.linspace(4.5, -4.5, 50)

                    cBd = iter(cm.Purples(np.linspace(0.3, 0.6, len(linsp))))

                    for lin in linsp:
                        c = next(cBd)
                        len(gap_thm3_.to_numpy())
                        add_something = np.add(gap_thm3_.to_numpy(), lin * to_be_added)
                        plt.plot(plt_iters[2:], add_something[2:], color=c)

                    plt.plot(plt_iters[2:], gap_[2:], label="MC approximation")

                    plt.xlabel(r"$N$")
                    plt.ylabel(str_plot)
                    plt.legend(loc='lower right')
                    plt.title(str_plot_name)
                    plt.savefig(fig_folder + 'fig_gap_comparison1overN' + str(N) + "_" + str(alpha) + "_" + str(dim) + "_" + str(loss) + "_" + str(
                        max_epochs) + "_" + str(epoch) + ".png")
                    plt.close()
