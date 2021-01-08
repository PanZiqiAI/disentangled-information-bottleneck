
import torch
from torch import nn


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class SpectralNorm(nn.Module):
    """
    Spectral Normalization.
    """
    def __init__(self, module, power_iter=1):
        super(SpectralNorm, self).__init__()
        # Config & Architecture
        self._power_iter = power_iter
        self._module = module
        # Make params
        self._make_params()

    @staticmethod
    def l2normalize(v):
        return v / (v.norm() + 1e-10)

    def _make_params(self):
        # Get weight
        w = getattr(self._module, 'weight')
        height, width = w.data.shape[0], w.view(w.data.shape[0], -1).data.shape[1]
        # Set buffers
        self._module.register_buffer('weight_u', w.data.new(height).normal_(0, 1))
        self._module.register_buffer('weight_v', w.data.new(width).normal_(0, 1))
        del self._module._parameters['weight']
        self._module.register_buffer('weight', w.data)
        # Set params
        self._module.register_parameter('weight_base', nn.Parameter(w.data, requires_grad=True))

    def _update_u_v(self):
        # Get u & v
        u = getattr(self._module, 'weight_u')
        v = getattr(self._module, 'weight_v')
        w_base = getattr(self._module, 'weight_base')
        # Approximate singular vectors corresponding to the largest singular value
        height = w_base.data.shape[0]
        for _ in range(self._power_iter):
            v.data = self.l2normalize(torch.mv(torch.t(w_base.view(height, -1).data), u.data))
            u.data = self.l2normalize(torch.mv(w_base.view(height, -1).data, v.data))
        # Get the largest singular value
        sigma = u.dot(w_base.view(height, -1).mv(v))
        # Set weight
        self._module.register_buffer('weight', w_base / sigma.expand_as(w_base))

    def forward(self, *args):
        # Spectral norm
        self._update_u_v()
        # Forward
        return self._module(*args)


def get_spectral_norm(module, spectral_norm=1):
    if spectral_norm:
        return SpectralNorm(module, power_iter=spectral_norm)
    else:
        return module


def get_sequential_layers(*layers):
    # Get valid layers
    module = []
    # Check each layer
    for layer in layers:
        if layer is None: continue
        module.append(layer)
    # Get module
    assert module
    return nn.Sequential(*module)

