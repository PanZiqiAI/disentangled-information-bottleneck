
import math
import torch


# ----------------------------------------------------------------------------------------------------------------------
# Operations
# ----------------------------------------------------------------------------------------------------------------------

def resampling(mu, std, **kwargs):
    """
    Resampling trick.
    """
    # Multi sampling. (batch*n_samples, nz)
    if 'n_samples' in kwargs.keys():
        if kwargs['n_samples'] > 0:
            eps = torch.randn(mu.size(0), kwargs['n_samples'], mu.size(1), device=mu.device)
            ret = eps.mul(std.unsqueeze(1) if isinstance(std, torch.Tensor) else std).add(mu.unsqueeze(1))
            return ret.reshape(-1, ret.size(2))
        else:
            return mu
    # Single sampling. (batch, nz)
    else:
        eps = torch.randn(mu.size(), device=mu.device)
        return eps.mul(std).add(mu)


def repeat(x, num):
    """
    :param x: (batch, ...)
    :param num:
    :return: (batch*num, ...)
    """
    x = x.unsqueeze(1).expand(x.size(0), num, *x.size()[1:])
    return x.reshape(-1, *x.size()[2:])


def clustering(mu, factor):
    """
    :param mu: (batch, nz)
    :param factor:
    :return: (batch, nz)
    """
    # 1. Calculate pairwise distance & mask. (batch, batch)
    distance = torch.mean((mu.unsqueeze(1) - mu.unsqueeze(0)).abs(), dim=2)
    mask = (distance < factor) * 1.0
    # 2. Averaging. (batch, batch, nz) -> (batch, nz)
    clustered_mu = torch.sum(mask.unsqueeze(-1) * mu.unsqueeze(0), dim=1) / mask.sum(dim=1).unsqueeze(-1)
    # Return
    return clustered_mu


# ----------------------------------------------------------------------------------------------------------------------
# Calculation
# ----------------------------------------------------------------------------------------------------------------------

def gaussian_kl_div(params1, params2='none', reduction='sum', average_batch=False):
    """
        0.5 * {
            sum_j [ log(var2)_j - log(var1)_j ]
            + sum_j [ (mu1 - mu2)^2_j / var2_j ]
            + sum_j (var1_j / var2_j)
            - K
        }
    :return:
    """
    assert reduction in ['sum', 'mean']
    # 1. Get params
    # (1) First
    mu1, std1 = params1
    # (2) Second
    if params2 == 'none':
        mu2 = torch.zeros(*mu1.size()).to(mu1.device)
        std2 = torch.ones(*std1.size()).to(std1.device)
    else:
        mu2, std2 = params2
    # 2. Calculate result
    result = 0.5 * (
        2 * (torch.log(std2) - torch.log(std1))
        + ((mu1 - mu2) / std2) ** 2
        + (std1 / std2) ** 2
        - 1)
    if reduction == 'sum':
        result = result.sum(dim=-1)
    else:
        result = result.mean(dim=-1)
    if average_batch:
        result = result.mean()
    # Return
    return result


def gaussian_log_density_marginal(sample, params, mesh=False):
    """
    Estimate Gaussian log densities:
        For not mesh:
            log p(sample_i|params_i), i in [batch]
        Otherwise:
            log p(sample_i|params_j), i in [num_samples], j in [num_params]
    :param sample: (num_samples, dims)
    :param params: mu, std. Each is (num_params, dims)
    :param mesh:
    :return:
        For not mesh: (num_sample, dims)
        Otherwise: (num_sample, num_params, dims)
    """
    # Get data
    mu, std = params
    # Mesh
    if mesh:
        sample = sample.unsqueeze(1)
        mu, std = mu.unsqueeze(0), std.unsqueeze(0)
    # Calculate
    # (1) log(2*pi)
    constant = math.log(2 * math.pi)
    # (2) 2 * log std_i
    log_det_std = 2 * torch.log(std)
    # (3) (x-mu)_i^2 / std_i^2
    dev_inv_std_dev = ((sample - mu) / std) ** 2
    # Get result
    log_prob_marginal = - 0.5 * (constant + log_det_std + dev_inv_std_dev)
    # Return
    return log_prob_marginal
