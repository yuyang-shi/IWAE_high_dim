import numpy as np
from functools import partial
import torch
from torch.distributions import Normal, Independent
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import hydra
from utils import logmeanexp, lognormexp
import utils


## Defining minus the VR-IWAE bound (reparameterized and doubly-reparameterized)
def _vr_iwae_loss_given_weights(log_ws, alpha=0):
    alpha_log_ws = log_ws * (1 - alpha)
    if alpha == 1:
        return - log_ws.mean(), alpha_log_ws
    else:
        return - logmeanexp(alpha_log_ws, dim=0).mean() / (1 - alpha), alpha_log_ws


def vr_iwae_loss(x, pz, qz_fn, N=1, alpha=0, **kwargs):
    qz = qz_fn(x)
    zs = qz.rsample((N,))
    log_ws = pz.log_prob(zs) - qz.log_prob(zs)
    return _vr_iwae_loss_given_weights(log_ws, alpha)[0], log_ws


def vr_iwae_dreg_loss(x, pz, qz_fn, N=1, alpha=0, encoder=None, qz_x_dist_fn=None):
    encode = encoder(x)
    qz_x = qz_x_dist_fn(encode)
    zs = qz_x.rsample((N,))
    qz_x_dist_detached = qz_x_dist_fn(encode.detach())
    zs_detached = zs.detach()
    log_ws_q_detached = pz.log_prob(zs_detached) - qz_x_dist_detached.log_prob(zs_detached)

    # theta objective (no phi grad)
    loss_theta, alpha_log_ws_q_detached = _vr_iwae_loss_given_weights(log_ws_q_detached, alpha)

    # phi objective (no theta grad)
    normalized_alpha_ws_detached = lognormexp(alpha_log_ws_q_detached.detach(), dim=0).exp()
    log_ws_q_dist_detached = pz.log_prob(zs) - qz_x_dist_detached.log_prob(zs)
    loss_phi = - ((alpha * normalized_alpha_ws_detached + (1 - alpha) * normalized_alpha_ws_detached ** 2) * (log_ws_q_dist_detached - log_ws_q_detached)).sum(dim=0).mean()

    return loss_theta + loss_phi, log_ws_q_detached


## Computing the gap using
# (i) the approximation of Thm3
# (ii) the approximation of Thm5
# (iii) a MC estimate of the VR-IWAE bound
def compute_gamma_alpha_LogNormal(alpha, B_d):
    gamma_alpha = 1/(1-alpha) * (np.exp((1-alpha)**2 * B_d ** 2)-1)
    return gamma_alpha


def compute_gap_1overN(gamma_alpha, alpha, B_d, J):
    results_loss = []
    for j in range(1, J):
        results_loss.append( - alpha * B_d **2 /2 - gamma_alpha / (2* 2**j))
    results_gap = np.array(results_loss)

    return results_gap


def compute_gap_LogNormal(B_d, alpha, J):
    results_loss = []
    for j in range(1, J):
        results_loss.append(-B_d ** 2 / 2 + B_d * np.sqrt(2 * np.log(2 ** j)) + np.log(j) / (alpha - 1))
    results_gap = np.array(results_loss)

    return results_gap


def compute_gap_MC_estimate(loss_name, data, pz, qz_fn, J, alpha, encoder, qz_x_dist_fn, nb_repeat=100):
    results_loss = []
    for j in range(1, J):
        results_loss_repeat = []
        for _ in range(nb_repeat):
            results_loss_repeat.append(loss_name(data, pz, qz_fn, N=2 ** j, alpha=alpha, encoder=encoder, qz_x_dist_fn=qz_x_dist_fn)[0].item())
        results_loss.append(np.mean(results_loss_repeat))
    results_gap = - np.array(results_loss)

    return results_gap


@hydra.main(config_path="./conf", config_name="log_normal_gaussian")
def main(args):
    dim = args.dim
    seed = args.seed
    N = args.N
    alpha = args.alpha
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    loss = args.loss
    if loss == 1:
        loss_name = vr_iwae_loss
    else:
        loss_name = vr_iwae_dreg_loss
    print(device)
    J = 10

    data = Normal(0, np.sqrt(2)).sample((1, dim)).to(device)

    true_mu_theta = torch.zeros(dim).to(device)
    true_mu_phi = true_mu_theta

    if seed is not None:
        utils.manual_seed(seed)

    mu_phi = torch.ones(dim).to(device)#torch.randn(dim, device=device)

    pz = Independent(Normal(true_mu_theta, 1), 1)
    qz_fn = lambda x: Independent(Normal(mu_phi, 1), 1)
    encoder = lambda x: mu_phi
    qz_x_dist_fn = lambda encode: Independent(Normal(encode, 1), 1)

    gap_estim = compute_gap_MC_estimate(loss_name, data, pz, qz_fn, J, alpha, encoder, qz_x_dist_fn)
    filename = "gap_MC_" + str(0) + ".npy"
    np.save(filename, gap_estim)

    mse = mean_squared_error(true_mu_phi.cpu(), mu_phi.detach().cpu())
    B_d = np.sqrt(dim * mse)
    gamma_alpha = compute_gamma_alpha_LogNormal(alpha, B_d)
    gap_1overN = compute_gap_1overN(gamma_alpha, alpha, B_d, 10)
    filename = "gap_1overN_" + str(0) + ".npy"
    np.save(filename, gap_1overN)

    gap_LogNormal = compute_gap_LogNormal(B_d, alpha, J)
    filename = "gap_LogNormal_" + str(0) + ".npy"
    np.save(filename, gap_LogNormal)

    # Training
    mu_phi.requires_grad_()

    freq = 100
    loss_fn = partial(loss_name, N=N, alpha=alpha)
    optimizer = utils.get_optimizer([mu_phi], args)

    pz = Independent(Normal(true_mu_theta, 1), 1)
    qz_x = lambda x: Independent(Normal(mu_phi, 1), 1)

    results_mse_mu_phi = []
    mse = mean_squared_error(true_mu_phi.cpu(), mu_phi.detach().cpu())
    results_mse_mu_phi.append(mse)

    for i in tqdm(range(1, args.max_epochs + 1)):
        optimizer.zero_grad()
        if i == 1 or i % freq == 0:
            mse = mean_squared_error(true_mu_phi.cpu(), mu_phi.detach().cpu())
            results_mse_mu_phi.append(mse)
            B_d = np.sqrt(dim * mse)

            gap_estim = compute_gap_MC_estimate(loss_name, data, pz, qz_fn, J, alpha, encoder, qz_x_dist_fn)
            filename = "gap_MC_" + str(i) + ".npy"
            np.save(filename, gap_estim)

            gamma_alpha = compute_gamma_alpha_LogNormal(alpha, B_d)
            gap_1overN = compute_gap_1overN(gamma_alpha, alpha, B_d, 10)
            filename = "gap_1overN_" + str(i) + ".npy"
            np.save(filename, gap_1overN)

            gap_LogNormal = compute_gap_LogNormal(B_d, alpha, J)
            filename = "gap_LogNormal_" + str(i) + ".npy"
            np.save(filename, gap_LogNormal)

            '''
            plt.figure()
            plt.plot(gap_estim)
            plt.plot(gap_thm3)
            plt.plot(gap_thm6, color='red')
            plt.show()
            '''

        loss, _ = loss_fn(data, pz, qz_x, encoder=encoder, qz_x_dist_fn=qz_x_dist_fn)

        loss.backward()
        optimizer.step()

    # Save
    np.save("mse_mu_phi.npy", results_mse_mu_phi)

    '''
    plt.style.use('bmh')
    plt.rcParams['figure.facecolor'] = '1'
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams["legend.loc"] = 'lower right'  # 'upper left'

    plt_iters = [1] + [i for i in range(1, args.max_epochs+1) if i % freq == 0]

    plt.figure()
    plt.plot(plt_iters, results_mse_mu_phi, label=str(loss_name))
    plt.legend(loc=0)
    plt.xlabel("iterations")
    plt.ylabel(r"$B_d^2$")
    plt.savefig("fig_mse_mu_phi.png")
    '''


if __name__ == '__main__':
    main()







