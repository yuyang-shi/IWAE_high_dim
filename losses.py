from utils import logmeanexp, lognormexp


def _vr_iwae_loss_given_weights_v1(log_ws, alpha=0):
    alpha_log_ws = log_ws * (1 - alpha)
    if alpha == 1:
        return - log_ws.mean(), alpha_log_ws
    else:
        return - logmeanexp(alpha_log_ws, dim=0).mean() / (1 - alpha), alpha_log_ws


def _vr_iwae_loss_given_weights_v2(log_ws, alpha=0):
    _vr_iwae_loss, alpha_log_ws = _vr_iwae_loss_given_weights_v1(log_ws, alpha)
    normalized_alpha_ws_detached = lognormexp(alpha_log_ws.detach(), dim=0).exp()
    vr_iwae_loss = - (normalized_alpha_ws_detached * log_ws).sum(dim=0).mean()
    return vr_iwae_loss - vr_iwae_loss.detach() + _vr_iwae_loss.detach(), alpha_log_ws, normalized_alpha_ws_detached


def vr_iwae_loss_v1(x, pz, px_z_fn, qz_x_fn, N=1, alpha=0, **kwargs):
    qz_x = qz_x_fn(x)
    zs = qz_x.rsample((N,))
    log_ws = pz.log_prob(zs) + px_z_fn(zs).log_prob(x) - qz_x.log_prob(zs)
    return _vr_iwae_loss_given_weights_v1(log_ws, alpha)[0], log_ws


def vr_iwae_loss_v2(x, pz, px_z_fn, qz_x_fn, N=1, alpha=0, **kwargs):
    qz_x = qz_x_fn(x)
    zs = qz_x.rsample((N,))
    log_ws = pz.log_prob(zs) + px_z_fn(zs).log_prob(x) - qz_x.log_prob(zs)
    return _vr_iwae_loss_given_weights_v2(log_ws, alpha)[0], log_ws


def vr_iwae_dreg_loss_v1(x, pz, px_z_fn, qz_x_fn, N=1, alpha=0, encoder=None, qz_x_dist_fn=None):
    encode = encoder(x)
    qz_x = qz_x_dist_fn(encode)
    zs = qz_x.rsample((N,))
    qz_x_dist_detached = qz_x_dist_fn(encode.detach())
    zs_detached = zs.detach()
    log_ws_q_detached = pz.log_prob(zs_detached) + px_z_fn(zs_detached).log_prob(x) - qz_x_dist_detached.log_prob(zs_detached)

    # theta objective (no phi grad)
    loss_theta, alpha_log_ws_q_detached = _vr_iwae_loss_given_weights_v1(log_ws_q_detached, alpha)

    # phi objective (no theta grad)
    normalized_alpha_ws_detached = lognormexp(alpha_log_ws_q_detached.detach(), dim=0).exp()
    log_ws_q_dist_detached = pz.log_prob(zs) + px_z_fn(zs).log_prob(x) - qz_x_dist_detached.log_prob(zs)
    loss_phi = - ((alpha * normalized_alpha_ws_detached + (1 - alpha) * normalized_alpha_ws_detached ** 2) * (log_ws_q_dist_detached - log_ws_q_detached)).sum(dim=0).mean()

    return loss_theta + loss_phi, log_ws_q_detached


def vr_iwae_dreg_loss_v2(x, pz, px_z_fn, qz_x_fn, N=1, alpha=0, encoder=None, qz_x_dist_fn=None):
    encode = encoder(x)
    qz_x = qz_x_dist_fn(encode)
    zs = qz_x.rsample((N,))
    qz_x_dist_detached = qz_x_dist_fn(encode.detach())
    zs_detached = zs.detach()
    log_ws_q_detached = pz.log_prob(zs_detached) + px_z_fn(zs_detached).log_prob(x) - qz_x_dist_detached.log_prob(zs_detached)
    
    # theta objective (no phi grad)
    loss_theta, _, normalized_alpha_ws_detached = _vr_iwae_loss_given_weights_v2(log_ws_q_detached, alpha)

    # phi objective (no theta grad)
    log_ws_q_dist_detached = pz.log_prob(zs) + px_z_fn(zs).log_prob(x) - qz_x_dist_detached.log_prob(zs)
    loss_phi = - ((alpha * normalized_alpha_ws_detached + (1 - alpha) * normalized_alpha_ws_detached ** 2) * (log_ws_q_dist_detached - log_ws_q_detached)).sum(dim=0).mean()

    return loss_theta + loss_phi, log_ws_q_detached


def vr_iwae_dreg_loss_v3(x, pz, px_z_fn, qz_x_fn, N=1, alpha=0, encoder=None, qz_x_dist_fn=None):
    encode = encoder(x)
    qz_x = qz_x_dist_fn(encode)
    zs = qz_x.rsample((N,))
    qz_x_dist_detached = qz_x_dist_fn(encode.detach())
    zs_detached = zs.detach()
    log_ws_q_detached = pz.log_prob(zs_detached) + px_z_fn(zs_detached).log_prob(x) - qz_x_dist_detached.log_prob(zs_detached)

    # theta objective (no phi grad)
    loss_theta, alpha_log_ws_q_detached = _vr_iwae_loss_given_weights_v1(log_ws_q_detached, alpha)

    # phi objective
    normalized_alpha_ws_detached = lognormexp(alpha_log_ws_q_detached.detach(), dim=0).exp()
    log_ws_q_dist_detached = pz.log_prob(zs) + px_z_fn(zs).log_prob(x) - qz_x_dist_detached.log_prob(zs)
    loss_phi = - ((alpha * normalized_alpha_ws_detached + (1 - alpha) * normalized_alpha_ws_detached ** 2) * log_ws_q_dist_detached).sum(dim=0).mean()

    return loss_theta, loss_phi, log_ws_q_detached


def vr_iwae_dreg_loss_v4(x, pz, px_z_fn, qz_x_fn, N=1, alpha=0, encoder=None, qz_x_dist_fn=None):
    encode = encoder(x)
    qz_x = qz_x_dist_fn(encode)
    zs = qz_x.rsample((N,))
    qz_x_dist_detached = qz_x_dist_fn(encode.detach())
    zs_detached = zs.detach()
    log_ws_q_detached = pz.log_prob(zs_detached) + px_z_fn(zs_detached).log_prob(x) - qz_x_dist_detached.log_prob(zs_detached)

    # theta objective (no phi grad)
    loss_theta, _, normalized_alpha_ws_detached = _vr_iwae_loss_given_weights_v2(log_ws_q_detached, alpha)

    # phi objective
    log_ws_q_dist_detached = pz.log_prob(zs) + px_z_fn(zs).log_prob(x) - qz_x_dist_detached.log_prob(zs)
    loss_phi = - ((alpha * normalized_alpha_ws_detached + (1 - alpha) * normalized_alpha_ws_detached ** 2) * log_ws_q_dist_detached).sum(dim=0).mean()

    return loss_theta, loss_phi, log_ws_q_detached


def vr_iwae_dreg_loss_v5(x, pz, px_z_fn, qz_x_fn, N=1, alpha=0, encoder=None, qz_x_dist_fn=None):
    encode = encoder(x)
    qz_x = qz_x_dist_fn(encode)
    zs = qz_x.rsample((N,))
    qz_x_dist_detached = qz_x_dist_fn(encode.detach())
    log_ws_q_dist_detached = pz.log_prob(zs) + px_z_fn(zs).log_prob(x) - qz_x_dist_detached.log_prob(zs)
    
    # theta objective
    loss_theta, alpha_log_ws_q_detached = _vr_iwae_loss_given_weights_v1(log_ws_q_dist_detached, alpha)

    # phi gradient correction
    normalized_alpha_ws_detached = lognormexp(alpha_log_ws_q_detached.detach(), dim=0).exp()
    zs.register_hook(lambda grad: (alpha + (1 - alpha) * normalized_alpha_ws_detached.unsqueeze(-1)) * grad)

    return loss_theta, log_ws_q_dist_detached


def vr_iwae_dreg_loss_v6(x, pz, px_z_fn, qz_x_fn, N=1, alpha=0, encoder=None, qz_x_dist_fn=None):
    encode = encoder(x)
    qz_x = qz_x_dist_fn(encode)
    zs = qz_x.rsample((N,))
    qz_x_dist_detached = qz_x_dist_fn(encode.detach())
    log_ws_q_dist_detached = pz.log_prob(zs) + px_z_fn(zs).log_prob(x) - qz_x_dist_detached.log_prob(zs)
    
    # theta objective
    loss_theta, _, normalized_alpha_ws_detached = _vr_iwae_loss_given_weights_v2(log_ws_q_dist_detached, alpha)

    # phi gradient correction
    zs.register_hook(lambda grad: (alpha + (1 - alpha) * normalized_alpha_ws_detached.unsqueeze(-1)) * grad)

    return loss_theta, log_ws_q_dist_detached