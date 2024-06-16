import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import cm
import seaborn as sns
import glob

# Parameters
alpha_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
dim_list = [20, 100, 1000]
perturb_sig_list = [0.01, 0.5]
figure_path = "figures_linear_gaussian"
loss_name = "rep"
num_run = 10
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
            if loss_name == 'rep':
                experiment_date = "2023-05-23" if alpha==1 else "2022-09-07" 
                if perturb_sig == 0.01:
                    glob_str = f"./experiments/linear_gaussian/{experiment_date}/cfg-alpha={alpha},dim={dim}/*/*/df_results.pkl"
                else:
                    glob_str = f"./experiments/linear_gaussian/{experiment_date}/cfg-alpha={alpha},dim={dim},perturb_sig={perturb_sig}/*/*/df_results.pkl"
            else:
                experiment_date = "2023-05-23" if alpha==1 else "2022-09-09" 
            
                if perturb_sig == 0.01:
                    glob_str = f"./experiments/linear_gaussian/{experiment_date}/cfg-alpha={alpha},dim={dim},loss_name=vr_iwae_dreg_loss_v6/*/*/df_results.pkl"
                else:
                    glob_str = f"./experiments/linear_gaussian/{experiment_date}/cfg-alpha={alpha},dim={dim},loss_name=vr_iwae_dreg_loss_v6,perturb_sig={perturb_sig}/*/*/df_results.pkl"
            
            experiment_files_list = sorted(glob.glob(glob_str))
            print(glob_str)
            print(experiment_files_list)

            assert len(experiment_files_list) == num_run
            for i in range(num_run):
                df_results = pd.read_pickle(experiment_files_list[i]).iloc[:9]
                df_results.index = 2 ** (df_results.index + 1)
                df_results = df_results.rename_axis("$N$")
                for key in ["vr_iwae", "vr_iwae_p_grad", "vr_iwae_q_grad"]:
                    df_results[f'{key}_snr'] = df_results[f'{key}_mean'].abs() / df_results[f'{key}_std']

                df_results_list.append(df_results)
            df_results_all.append(pd.concat(df_results_list, keys=list(range(10)), names=["run"]))
        df_results_all = pd.concat(df_results_all, keys=alpha_list, names=["alpha"])


        # # Plot true_vr_bound vs alpha
        # df_results_all.loc[pd.IndexSlice[:, :, 2], :].groupby(level=0).agg({'true_vr_bound': ['mean', 'std']})['true_vr_bound'].plot(y="mean", title=r"True VR bound ($d=$" + str(dim) + ", $\sigma_{\mathrm{perturb}}=$" + str(perturb_sig) + ")", legend=False, xlabel=r"$\alpha$", ylabel=r"$\mathcal{L}^{(\alpha)}(\theta, \phi; x)$")
        # # sns.regplot(x="alpha", y="true_vr_bound", data=df_results_all.loc[pd.IndexSlice[:, :, 0], :]['true_vr_bound'].reset_index(), x_estimator=np.mean)
        # plt.tight_layout()
        # plt.savefig(os.path.join(figure_path, f"true_vr_bound_against_alpha_dim_{dim}_perturb_sig_{perturb_sig}.png"))
        # plt.close()

        # # # Plot vr_iwae vs alpha
        # # df_results_all.groupby(level=[0,2]).agg({'vr_iwae_mean': ['mean', 'std']}).unstack(level=1)["vr_iwae_mean"].plot(y="mean", title=r"VR-IWAE bound ($d=$" + str(dim) + ", $\sigma_{\mathrm{perturb}}=$" + str(perturb_sig) + ")", xlabel=r"$\alpha$", ylabel=r"$\ell^{(\alpha)}_{N,d}(\theta, \phi; x)$")
        # # plt.plot(df_results_all.loc[pd.IndexSlice[:, :, 1], :].groupby(level=0).agg({'true_vr_bound': ['mean', 'std']})['true_vr_bound']['mean'], label=r"$\infty$")
        # # plt.legend()
        # # plt.tight_layout()
        # # plt.savefig(os.path.join(figure_path, f"vr_iwae_against_alpha_dim_{dim}_perturb_sig_{perturb_sig}.png"))
        # # plt.close()

        # # Plot vr_iwae vs N
        # df_results_all.groupby(level=[0,2]).agg({'vr_iwae_mean': ['mean', 'std']}).unstack(level=0)["vr_iwae_mean"].plot(y="mean", title=r"VR-IWAE bound ($d=$" + str(dim) + ", $\sigma_{\mathrm{perturb}}=$" + str(perturb_sig) + ")", xlabel=r"$N$", ylabel=r"$\ell^{(\alpha)}_{N,d}(\theta, \phi; x)$", logx=True, colormap='coolwarm')
        # plt.legend(labels=[rf'$\alpha={alpha}$' for alpha in alpha_list], loc='upper left')
        # plt.tight_layout()
        # plt.savefig(os.path.join(figure_path, f"vr_iwae_against_N_dim_{dim}_perturb_sig_{perturb_sig}.png"))
        # plt.close()

        # # Plot variational gap vs N
        # df_results_all['vr_iwae_gap'] = df_results_all["true_ll"] - df_results_all["vr_iwae_mean"]
        # df_results_all.groupby(level=[0,2]).agg({'vr_iwae_gap': ['mean', 'std']}).unstack(level=0)["vr_iwae_gap"].plot(y="mean", title=r"Variational gap ($d=$" + str(dim) + ", $\sigma_{\mathrm{perturb}}=$" + str(perturb_sig) + ")", xlabel=r"$N$", ylabel=r"$\Delta^{(\alpha)}_{N,d}(\theta, \phi; x)$", logx=True, logy=True, colormap='coolwarm')
        # plt.autoscale(False)
        # plt.loglog(2**np.arange(1, J), df_results_all['vr_iwae_gap'].groupby(level=[0,2]).mean().loc[pd.IndexSlice[0, 2]] / 2**np.arange(J-1), ls=':', c='k')
        # plt.legend(labels=[rf'$\alpha={alpha}$' for alpha in alpha_list] + ["$\Theta(1/N)$"], loc='upper left')
        # plt.tight_layout()
        # plt.savefig(os.path.join(figure_path, f"vr_iwae_gap_against_N_dim_{dim}_perturb_sig_{perturb_sig}.png"))
        # plt.close()
        # plt.autoscale(True)
        
        # Plot curves
        # for alpha in [0,0.2,0.5]:
        #     for i in range(datapoints):
        #         B_d = df_results_all.loc[pd.IndexSlice[alpha, 0, 2], f"B_d_datapoints{i}"]
        #         vr_iwae_means = df_results_all.loc[pd.IndexSlice[alpha, 0, :], f"vr_iwae_mean_datapoints{i}"].to_numpy()
        #         vr_iwae_1overN = df_results_all.loc[pd.IndexSlice[alpha, 0, :], f"vr_iwae_bound_1overN_datapoints{i}"].to_numpy()
        #         #vr_iwae_estim_1overN = df_results_all.loc[pd.IndexSlice[alpha, 0, :], f"vr_iwae_bound_1overN_LogNormal_datapoints{i}"].to_numpy()
        #         vr_iwae_ApproxLogNormal = df_results_all.loc[pd.IndexSlice[alpha, 0, :], f"vr_iwae_bound_ApproxLogNormal_datapoints{i}"].to_numpy()
        #         vr_iwae_LogNormal = df_results_all.loc[pd.IndexSlice[alpha, 0, :], f"vr_iwae_bound_LogNormal_datapoints{i}"].to_numpy()
        #         plt_iters = [2 ** j for j in range(1, J)]

        #         # 1overN plots
        #         to_be_added = np.array([1 / j for j in plt_iters])
        #         linsp = np.linspace(4.5, -4.5, 50)
        #         cBd1 = iter(cm.Purples(np.linspace(0.3, 0.6, len(linsp))))
        #         cBd = iter(cm.Greens(np.linspace(0.3, 0.6, len(linsp))))

        #         plt.figure()
        #         for lin in linsp:
        #             c1 = next(cBd1)
        #             add_something_estim = np.add(vr_iwae_1overN, lin * to_be_added)
        #             plt.plot(plt_iters, add_something_estim, color=c1)

        #         plt.plot(plt_iters, vr_iwae_means, label="MC approximation")
        #         plt.legend(loc='lower right')
        #         plt.xlabel(r"$N$")
        #         plt.ylabel(r"$\ell^{(\alpha)}_{N,d}(\theta, \phi; x)$")
        #         plt.title(r"VR-IWAE bound ($\alpha =$" + str(alpha) + "$, d =$" + str(dim) + ", $\sigma_{\mathrm{perturb}}=$" + str(perturb_sig) + ")")
        #         plt.tight_layout()
        #         plt.savefig(figure_path + '/vr_iwae_bound_against_N_1overN_' + str(dim) + "_perturb_sig_" + str(perturb_sig) + "_" + str(alpha) + 'datapoints' + str(i) +'.png')
        #         plt.close()

        #         # plt.figure()
        #         # for lin in linsp:
        #         #     c = next(cBd)
        #         #     add_something = np.add(vr_iwae_estim_1overN, lin * to_be_added)
        #         #     plt.plot(plt_iters, add_something,color=c)

        #         # plt.plot(plt_iters, vr_iwae_means, label="MC approximation")
        #         # plt.legend(loc='lower right')
        #         # plt.xlabel(r"$N$")
        #         # plt.ylabel(r"$\ell^{(\alpha)}_{N,d}(\theta, \phi; x)$")
        #         # plt.title(r"VR-IWAE bound ($\alpha =$" + str(alpha) + "$, d =$" + str(dim) + ")")
        #         # plt.savefig(
        #         #     figure_path + '/vr_iwae_bound_against_N_1overN_LogNormal_' + str(dim) + "_perturb_sig_" + str(perturb_sig) + "_" + str(alpha) + 'datapoints' + str(
        #         #         i) + '.png')
        #         # plt.close()

        #         # Approximated Log-normal plots
        #         to_be_added = np.array([np.sqrt(dim) * np.log(np.log(j)) / np.sqrt(np.log(j)) for j in plt_iters])
        #         linsp = np.linspace(-2, 0.5, 50)
        #         cBd = iter(cm.Greens_r(np.linspace(0.6, 0.9, len(linsp))))

        #         plt.figure()
        #         for lin in linsp:
        #             c = next(cBd)
        #             add_something = np.add(vr_iwae_ApproxLogNormal, lin * to_be_added)
        #             plt.plot(plt_iters[2:], add_something[2:], color=c)

        #         plt.plot(plt_iters, vr_iwae_means, label="MC approximation")
        #         plt.legend(loc='lower right')
        #         plt.xlabel(r"$N$")
        #         plt.ylabel(r"$\ell^{(\alpha)}_{N,d}(\theta, \phi; x)$")
        #         plt.title(r"VR-IWAE bound ($\alpha =$" + str(alpha) + "$, d =$" + str(dim) + ", $\sigma=$" + str(perturb_sig) + ")")
        #         plt.tight_layout()
        #         plt.savefig(os.path.join(figure_path,
        #                     'vr_iwae_bound_against_N_ApproxLogNormal_' + str(dim) + "_perturb_sig_" + str(perturb_sig) + "_" + str(
        #                     alpha) + 'datapoints' + str(i) + '.png'))
        #         plt.close()

        #         # Log-normal plots
        #         to_be_added = np.array([B_d * np.log(np.log(j)) / np.sqrt(np.log(j)) for j in plt_iters])
        #         linsp = np.linspace(-2, 0.5, 50)
        #         cBd = iter(cm.Reds_r(np.linspace(0.6, 0.9, len(linsp))))

        #         plt.figure()
        #         for lin in linsp:
        #             c = next(cBd)
        #             add_something = np.add(vr_iwae_LogNormal, lin * to_be_added)
        #             plt.plot(plt_iters[2:], add_something[2:], color=c)

        #         plt.plot(plt_iters, vr_iwae_means, label="MC approximation")
        #         plt.legend(loc='lower right')
        #         plt.xlabel(r"$N$")
        #         plt.ylabel(r"$\ell^{(\alpha)}_{N,d}(\theta, \phi; x)$")
        #         plt.title(r"VR-IWAE bound ($\alpha =$" + str(alpha) + "$, d =$" + str(dim) + ", $\sigma_{\mathrm{perturb}}=$" + str(perturb_sig) + ")")
        #         plt.tight_layout()
        #         plt.savefig(os.path.join(figure_path, 
        #                     'vr_iwae_bound_against_N_LogNormal_' + str(dim) + "_perturb_sig_" + str(perturb_sig) + "_" + str(
        #                     alpha) + 'datapoints' + str(i) + '.png'))
        #         plt.close()

        # # Plot var vs N
        # for target_type in ['_p_grad', '_q_grad']:
        #     df_results_all[f'vr_iwae{target_type}_std'].pow(2).groupby(level=[0,2]).mean().unstack(level=0).plot(logy=True, title=f"vr_iwae{target_type}_var", xlabel=r"$N$", ylabel="Var", logx=True)
        #     plt.savefig(os.path.join(figure_path, f"vr_iwae{target_type}_var_against_N_{loss_name + '_' if target_type == '_q_grad' else ''}dim_{dim}_perturb_sig_{perturb_sig}.png"))
        #     plt.close()

        # Plot snr vs N
        titles = [r"SNR: $\theta$ gradient ($d =$" + str(dim) + ", $\sigma_{\mathrm{perturb}}=$" + str(perturb_sig) + ")", r"SNR: $\phi$ gradient ($d =$" + str(dim) + ", $\sigma_{\mathrm{perturb}}=$" + str(perturb_sig) + ")"]
        for i, target_type in enumerate(['_p_grad', '_q_grad']):
            fig, ax = plt.subplots()
            df_results_all[f'vr_iwae{target_type}_snr'].groupby(level=[0,2]).mean().unstack(level=0).plot(logy=True, title=titles[i], xlabel=r"$N$", ylabel="SNR", logx=True, colormap='coolwarm', ax=ax)
            if target_type == '_p_grad':
                plt.loglog(2**np.arange(1, J), df_results_all[f'vr_iwae{target_type}_snr'].groupby(level=2).mean().loc[pd.IndexSlice[2]] * 2**(np.arange(J-1)/2), ls='-.', c='k')
                plt.legend(labels=[rf'$\alpha={alpha}$' for alpha in alpha_list] + ["$\Theta(\sqrt{N})$"], loc='upper left')
            elif loss_name == "drep":
                plt.legend(labels=[rf'$\alpha={alpha}$' for alpha in alpha_list], loc='upper left')
            else:
                plt.loglog(2**np.arange(1, J), df_results_all[f'vr_iwae{target_type}_snr'].groupby(level=2).mean().loc[pd.IndexSlice[2]] * 2**(np.arange(J-1)/2), ls='-.', c='k', label="theo1")
                plt.loglog(2**np.arange(1, J), df_results_all[f'vr_iwae{target_type}_snr'].groupby(level=2).mean().loc[pd.IndexSlice[2]] * 2**(-np.arange(J-1)/2), ls=':', c='k', label="theo2")
                handles, labels = ax.get_legend_handles_labels()
                print(handles, labels)
                # raise NotImplementedError
                legend1 = ax.legend(handles=handles[:-2], labels=[rf'$\alpha={alpha}$' for alpha in alpha_list], loc='upper left', ncol=2)
                legend2 = ax.legend(handles=handles[-2:], labels=["$\Theta(\sqrt{N})$", "$\Theta(1/\sqrt{N})$"], loc='lower left', ncol=2)         
                ax.add_artist(legend1)
                ax.add_artist(legend2)
            plt.tight_layout()
            plt.savefig(os.path.join(figure_path, f"vr_iwae{target_type}_snr_against_N_{loss_name + '_' if target_type == '_q_grad' else ''}dim_{dim}_perturb_sig_{perturb_sig}.png"))
            plt.close()

        # # Plot bias squared vs N
        # target_loss = "true_ll"
        # df_results_all['true_ll_q_grad'] = 0
        # titles = [r"MSE: VR-IWAE ($d =$" + str(dim) + ", $\sigma_{\mathrm{perturb}}=$" + str(perturb_sig) + ")", r"MSE: $\theta$ gradient ($d =$" + str(dim) + ", $\sigma_{\mathrm{perturb}}=$" + str(perturb_sig) + ")", r"MSE: $\phi$ gradient ($d =$" + str(dim) + ", $\sigma_{\mathrm{perturb}}=$" + str(perturb_sig) + ")"]
        # for i, target_type in enumerate(['', '_p_grad', '_q_grad']):
        #     df_results_all[f'vr_iwae_{target_loss}{target_type}_bias_squared'] = (df_results_all[f'vr_iwae{target_type}_mean'] - df_results_all[f'{target_loss}{target_type}']) ** 2
        #     # df_results_all[f'vr_iwae_{target_loss}{target_type}_bias_squared'].groupby(level=[0,2]).mean().unstack(level=0).plot(logy=True, title=f'vr_iwae_{target_loss}{target_type}_bias_squared', xlabel=r"$N$", ylabel="Bias Squared", logx=True, )
        #     # plt.savefig(os.path.join(figure_path, f"vr_iwae{target_type}_bias_squared_against_N_{loss_name + '_' if target_type == '_q_grad' else ''}dim_{dim}_perturb_sig_{perturb_sig}.png"))
        #     # plt.close()

        #     (df_results_all[f'vr_iwae_{target_loss}{target_type}_bias_squared'] + df_results_all[f'vr_iwae{target_type}_std'].pow(2)).groupby(level=[0,2]).mean().unstack(level=0).plot(logy=True, title=titles[i], xlabel=r"$N$", logx=True, ylabel="MSE", colormap='coolwarm')
        #     plt.legend(labels=[rf'$\alpha={alpha}$' for alpha in alpha_list], loc='lower left')
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(figure_path, f"vr_iwae{target_type}_mse_against_N_{loss_name + '_' if target_type == '_q_grad' else ''}dim_{dim}_perturb_sig_{perturb_sig}.png"))
        #     plt.close()


    # # fig, ax = plt.subplots()
    # # df_results_all.loc[0].groupby(level=1).mean()['vr_iwae_p_grad_snr'].plot(label='vr_iwae', logy=True, ax=ax)
    # # # df_results_all.loc[0].groupby(level=1).agg({'vr_iwae_p_grad_snr': ['mean', 'std']})['vr_iwae_p_grad_snr'].plot(y='mean', yerr='std', label='vr_iwae', logy=True, ax=ax)
    # # df_results_all.loc[0]['vr_iwae_p_grad_snr'].unstack(level=0).plot(label='vr_iwae', logy=True, ax=ax, legend=False, c='C0', alpha=0.3)
    # # plt.savefig(os.path.join(figure_path, ".png"))