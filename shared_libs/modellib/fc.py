
import torch
import torch.nn as nn
import torch.nn.functional as F
from shared_libs.utils.operations import resampling
from shared_libs.custom_packages.custom_pytorch.layers import get_spectral_norm


def init_weights(layer):
    """
    Initialize weights.
    """
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
        if layer.bias is not None: layer.bias.data.zero_()


########################################################################################################################
# Modules
########################################################################################################################

class FcBlock(nn.Module):
    """
    Class Discriminator Module.
    """
    def __init__(self, in_dim, hidden_dims, out_dim, act=nn.LeakyReLU(), use_dropout=True, use_bias=True):
        super(FcBlock, self).__init__()
        # Architecture
        # 1. Build
        # (1) Init
        layers = []
        # (2) Generate each layer
        for index, (ipt, opt) in enumerate(zip([in_dim] + hidden_dims, hidden_dims + [out_dim])):
            layers.append(nn.Linear(ipt, opt, bias=use_bias))
            if index < len(hidden_dims):
                layers.append(act)
                if use_dropout: layers.append(nn.Dropout())
        # (3) Generate sequential module
        self._block = nn.Sequential(*layers)

    def forward(self, embedding):
        """
        :param embedding: (N, C)
        :return:
        """
        return self._block(embedding)


# ----------------------------------------------------------------------------------------------------------------------
# Shared: Decoder
# ----------------------------------------------------------------------------------------------------------------------

class FCDecoder(nn.Module):
    """
    FC decoder.
    """
    def __init__(self, input_dim, hidden_dims, out_dim):
        super(FCDecoder, self).__init__()
        # 1. Architecture
        self._fc = FcBlock(input_dim, hidden_dims, out_dim, act=nn.ReLU(True), use_dropout=False)
        # 2. Init
        self.apply(init_weights)

    def forward(self, emb):
        return self._fc(emb)


# ----------------------------------------------------------------------------------------------------------------------
# IB Lagrangian: Encoders
# ----------------------------------------------------------------------------------------------------------------------

class VIBEncoder(nn.Module):
    """
    Deep Variational Information Bottleneck (VIB) encoder.
    """
    def __init__(self, input_dim, hidden_dims, emb_dim, softplus_scalar):
        super(VIBEncoder, self).__init__()
        # Config
        self._emb_dim = emb_dim
        self._softplus_scalar = softplus_scalar
        # 1. Architecture
        self._fc = FcBlock(input_dim, hidden_dims, emb_dim * 2, act=nn.ReLU(True), use_dropout=False)
        # 2. Init
        self.apply(init_weights)

    def forward(self, x):
        # 1. Get params
        params = self._fc(x)
        mu, log_std = torch.split(params, self._emb_dim, dim=1)
        #   Applying softplus for std
        if self._softplus_scalar > 0.0:
            std = F.softplus(log_std - self._softplus_scalar, beta=1)
        else:
            std = log_std.exp()
        #   Set params
        setattr(self, 'params', (mu, std))
        # 2. Resampling
        emb = resampling(mu, std)
        # Return
        return emb


class NIBEncoder(nn.Module):
    """
    Nonlinear Information Bottleneck (NIB) encoder.
    """
    def __init__(self, input_dim, hidden_dims, emb_dim, log_std, log_std_trainable):
        super(NIBEncoder, self).__init__()
        # Config
        self._emb_dim = emb_dim
        # 1. Architecture
        # (1) Fc to produce mu
        self._fc_mu = FcBlock(input_dim, hidden_dims, emb_dim, act=nn.ReLU(True), use_dropout=False)
        # (2) Log_std
        if log_std_trainable:
            self.register_parameter('log_std', torch.nn.Parameter(torch.FloatTensor([log_std]), requires_grad=True))
        else:
            self.register_buffer('log_std', torch.FloatTensor([log_std]))
        # 2. Init
        self.apply(init_weights)

    def forward(self, x):
        # 1. Get params
        mu = self._fc_mu(x)
        std = self.log_std.exp().expand(*mu.size())
        #   Set params
        setattr(self, 'params', (mu, std))
        # 2. Resampling
        emb = resampling(mu, std)
        # Return
        return emb


# ----------------------------------------------------------------------------------------------------------------------
# Disentangled IB: Encoder, reconstructor, density estimator
# ----------------------------------------------------------------------------------------------------------------------

class DisenIBEncoder(nn.Module):
    """
    Disentangled Information Bottleneck encoder.
    """
    def __init__(self, input_dim, hidden_dims, emb_dim):
        super(DisenIBEncoder, self).__init__()
        # Config
        self._emb_dim = emb_dim
        # 1. Architecture
        self._fc = FcBlock(input_dim, hidden_dims, emb_dim, act=nn.ReLU(True), use_dropout=False)
        # 2. Init
        self.apply(init_weights)

    def forward(self, x):
        return self._fc(x)


class FCReconstructor(nn.Module):
    """
    FC reconstructor module.
    """
    def __init__(self, style_dim, class_dim, hidden_dims, out_dim, num_classes):
        super(FCReconstructor, self).__init__()
        # 1. Architecture
        self.register_parameter('word_dict', torch.nn.Parameter(torch.randn(size=(num_classes, class_dim))))
        self._fc = FcBlock(style_dim + class_dim, hidden_dims, out_dim, act=nn.LeakyReLU(0.2), use_dropout=False)
        # 2. Init
        self.apply(init_weights)

    def forward(self, style_emb, class_label):
        # Get class emb
        class_emb = torch.index_select(self.word_dict, dim=0, index=class_label)
        # Reconstruction
        x = torch.cat((style_emb, class_emb), dim=1)
        return self._fc(x)


class DensityEstimator(nn.Module):
    """
    Estimating probability density.
    """
    def __init__(self, style_dim, class_dim):
        super(DensityEstimator, self).__init__()
        # 1. Architecture
        # (1) Pre-fc
        self._fc_style = get_spectral_norm(nn.Linear(in_features=style_dim, out_features=128, bias=True))
        self._fc_class = get_spectral_norm(nn.Linear(in_features=class_dim, out_features=128, bias=True))
        # (2) FC blocks
        self._fc_blocks = nn.Sequential(
            # Layer 1
            get_spectral_norm(nn.Linear(in_features=256, out_features=256, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            get_spectral_norm(nn.Linear(in_features=256, out_features=256, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            get_spectral_norm(nn.Linear(in_features=256, out_features=1, bias=True)))
        # 2. Init weights
        self.apply(init_weights)

    def _call_method(self, style_emb, class_emb):
        style_emb = self._fc_style(style_emb)
        class_emb = self._fc_class(class_emb)
        return self._fc_blocks(torch.cat([style_emb, class_emb], dim=1))

    def forward(self, style_emb, class_emb, mode):
        assert mode in ['orig', 'perm']
        # 1. q(s, t)
        if mode == 'orig':
            return self._call_method(style_emb, class_emb)
        # 2. q(s)q(t)
        else:
            # Permutation
            style_emb_permed = style_emb[torch.randperm(style_emb.size(0)).to(style_emb.device)]
            class_emb_permed = class_emb[torch.randperm(class_emb.size(0)).to(class_emb.device)]
            return self._call_method(style_emb_permed, class_emb_permed)
