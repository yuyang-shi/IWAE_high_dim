import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from matplotlib.pyplot import cm
from scipy.optimize import curve_fit

# Plot settings
plt.style.use('bmh')
plt.rcParams['figure.facecolor'] = '1'
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams["legend.loc"] = 'lower right'


def aux_plot_gap(N, max_epochs, params_folder, freq=100, J=10):

    nb_curves = [0] + [i for i in range(1, max_epochs + 1) if i % freq == 0]
    nb_curves = [i for i in range(1, max_epochs + 1) if i % freq == 0]

    color_j = iter(cm.Blues_r(np.linspace(0.3, 0.7, len(nb_curves))))
    plt_iters = [2 ** j for j in range(1, J + 1)]

    for j in nb_curves:
        c_j = next(color_j)
        experiments_folder = './experiments/linear_gaussian/**/' + params_folder + '/**/' + 'gap' + str(j) + '.npy'
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

        plt.plot(plt_iters, -df['mean'], label='it= ' + str(j), c=c_j)
        plt.fill_between(plt_iters, -df['q10'], -df['q90'], alpha=0.2, color=c_j)

        df = pd.DataFrame(None)


def aux_plot_gap_alpha(N, max_epochs, params_folder, freq=100, J=10, alpha=0., color='red'):

    nb_curves = [0] + [i for i in range(1, max_epochs + 1) if i % freq == 0]
    nb_curves = [i for i in range(1, max_epochs + 1) if i % freq == 0]

    color_j = iter(cm.Blues_r(np.linspace(0.3, 0.7, len(nb_curves))))
    plt_iters = [2 ** j for j in range(1, J + 1)]

    for j in nb_curves:
        c_j = next(color_j)
        experiments_folder = './experiments/linear_gaussian/**/' + params_folder + '/**/' + 'gap' + str(j) + '.npy'
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

        plt.plot(plt_iters, -df['mean'], label='alpha= ' + str(alpha))
        plt.fill_between(plt_iters, -df['q90'], -df['q10'], alpha=0.2)

        df = pd.DataFrame(None)


def aux_plot(loss_name_str, N, max_epochs, params_folder, freq, str_plot):
    # Fetch the files in the correct folder
    experiments_folder = './experiments/linear_gaussian/**/' + params_folder + '/**/'+ str_plot + '.npy'
    files_retrieved = glob.glob(experiments_folder)
    nb_run = len(files_retrieved)

    # Put the file retrieved inside a df
    filename = files_retrieved[0]
    file_content = np.load(filename)
    df = pd.DataFrame({'Run 1': file_content})

    for i in range(nb_run):
        filename = files_retrieved[i]
        file_content = np.load(filename)
        run_nb = 'Run ' + str(i+1)
        df[run_nb] = file_content.tolist()

    # Compute mean and quantiles
    df['mean'] = df.mean(axis=1)
    df['q10'] = df.quantile(q=0.10, axis=1)
    df['q90'] = df.quantile(q=0.90, axis=1)

    # Plot the final result
    plt_iters = [1] + [i for i in range(1, max_epochs + 1) if i % freq == 0]
    plt.plot(plt_iters, df['mean'], label='N = ' + str(N) + ", alpha =" + str_alpha)
    plt.fill_between(plt_iters, df['q10'], df['q90'], alpha=0.2)


if __name__ == '__main__':

    # Parameters
    # TODO: The parameters are set manually for now
    alpha_list = [0., 0.2, 0.5, 0.8]#[0., 0.2, 0.5, 0.8]
    dim = 100
    N = 32
    loss_name_str = "vr_iwae_loss_v2"
    max_epochs = 1000
    freq = 10

    for alpha in alpha_list:

        if alpha == 0.:
            str_alpha = "0."
        else:
            str_alpha = str(alpha)
        params_folder = 'cfg-N=' + str(N) + ',alpha=' + str_alpha + ',dim=' + str(dim) + ',loss_name=' + loss_name_str

        # Plot MSEs
        str_plot = 'nll'
        str_plot_name = r'Negative log-likelihood per iteration $(N =$' + str(N) + r'$, d=$' + str(dim) + ')'
        aux_plot(loss_name_str, N, max_epochs, params_folder, freq, str_plot)


    plt.xlabel("iterations")
    plt.ylabel(str_plot)
    plt.legend(loc='upper left')
    plt.title(str_plot_name)
    plt.show()


    for alpha in alpha_list:

        if alpha == 0.:
            str_alpha = "0."
        else:
            str_alpha = str(alpha)
        params_folder = 'cfg-N=' + str(N) + ',alpha=' + str_alpha + ',dim=' + str(dim) + ',loss_name=' + loss_name_str

        # Plot the variational gap
        aux_plot_gap_alpha(N, max_epochs, params_folder, freq=1000, alpha=alpha, color='red')

    plt.xlabel("N")
    plt.ylabel("variational gap")
    plt.legend(loc='lower right')
    plt.title(r'Variational gap as a function of $N$ and for $\alpha$ ' + r'(training: $N =$' + str(N) + r'$, d=$' + str(dim) + ')')
    plt.show()

