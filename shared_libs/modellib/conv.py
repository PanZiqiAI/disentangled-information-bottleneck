
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(layer):
    """
    Initialize weights.
    """
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        if layer.bias is not None: layer.bias.data.zero_()


########################################################################################################################
# Modules
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Shared
# ----------------------------------------------------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Decoder module.
    """
    def __init__(self, class_dim, num_classes):
        super(Decoder, self).__init__()
        # 1. Architecture
        self._fc = nn.Linear(in_features=class_dim, out_features=num_classes, bias=False)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, emb):
        return self._fc(emb)


class DensityEstimator(nn.Module):
    """
    Estimating probability density.
    """
    def __init__(self, style_dim, class_dim):
        super(DensityEstimator, self).__init__()
        # 1. Architecture
        # (1) Pre-fc
        self._fc_style = nn.Linear(in_features=style_dim, out_features=128, bias=True)
        self._fc_class = nn.Linear(in_features=class_dim, out_features=128, bias=True)
        # (2) FC blocks
        self._fc_blocks = nn.Sequential(
            # Layer 1
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            nn.Linear(in_features=256, out_features=1, bias=True))
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


# ----------------------------------------------------------------------------------------------------------------------
# MNIST
# ----------------------------------------------------------------------------------------------------------------------

class EncoderMNIST(nn.Module):
    """
    Encoder Module.
    """
    def __init__(self, nz):
        super(EncoderMNIST, self).__init__()
        # 1. Architecture
        # (1) Convolution
        self._conv_blocks = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.ReLU(inplace=True))
        # (2) FC
        self._fc = nn.Linear(in_features=256, out_features=nz, bias=True)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, x):
        x = self._conv_blocks(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        ret = self._fc(x)
        # Return
        return ret


class ReconstructorMNIST(nn.Module):
    """
    Decoder Module.
    """
    def __init__(self, style_dim, class_dim, num_classes):
        super(ReconstructorMNIST, self).__init__()
        # 1. Architecture
        self.register_parameter('word_dict', torch.nn.Parameter(torch.randn(size=(num_classes, class_dim))))
        # (1) FC
        self._fc_style = nn.Linear(in_features=style_dim, out_features=256, bias=True)
        self._fc_class = nn.Linear(in_features=class_dim, out_features=256, bias=True)
        # (2) Convolution
        self._deconv_blocks = nn.Sequential(
            # Layer 1
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid())
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, style_emb, class_label):
        # Get class dim
        class_emb = torch.index_select(self.word_dict, dim=0, index=class_label)
        # 1. FC
        style_emb = F.leaky_relu_(self._fc_style(style_emb), negative_slope=0.2)
        class_emb = F.leaky_relu_(self._fc_class(class_emb), negative_slope=0.2)
        # 2. Convolution
        x = torch.cat((style_emb, class_emb), dim=1)
        x = x.view(x.size(0), 128, 2, 2)
        x = self._deconv_blocks(x)
        # Return
        return x


class DiscriminatorMNIST(nn.Module):
    """
    Discriminator Module.
    """
    def __init__(self):
        super(DiscriminatorMNIST, self).__init__()
        # 1. Architecture
        # (1) Convolution
        self._conv_blocks = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=128, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # (2) FC
        self._fc = nn.Linear(in_features=512, out_features=2, bias=True)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, x):
        # 1. Convolution
        x = self._conv_blocks(x)
        # 2. FC
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self._fc(x)
        # Return
        return x

# ----------------------------------------------------------------------------------------------------------------------
# Shapes
# ----------------------------------------------------------------------------------------------------------------------


class EncoderShapes(nn.Module):
    """
    Encoder Module.
    """
    def __init__(self, nz):
        super(EncoderShapes, self).__init__()
        # 1. Architecture
        # (1) Convolution
        self._conv_blocks = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, bias=True),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.ReLU(inplace=True))
        # (2) FC
        self._fc = nn.Linear(in_features=256, out_features=nz, bias=True)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, x):
        x = self._conv_blocks(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        ret = self._fc(x)
        # Return
        return ret


class ReconstructorShapes(nn.Module):
    """
    Decoder Module.
    """
    def __init__(self, style_dim, class_dim, num_classes):
        super(ReconstructorShapes, self).__init__()
        # 1. Architecture
        self.register_parameter('word_dict', torch.nn.Parameter(torch.randn(size=(num_classes, class_dim))))
        # (1) FC
        self._fc_style = nn.Linear(in_features=style_dim, out_features=256, bias=True)
        self._fc_class = nn.Linear(in_features=class_dim, out_features=256, bias=True)
        # (2) Convolution
        self._deconv_blocks = nn.Sequential(
            # Layer 1
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=8, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 4
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=8, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 5
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid())
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, style_emb, class_label):
        # Get class dim
        class_emb = torch.index_select(self.word_dict, dim=0, index=class_label)
        # 1. FC
        style_emb = F.leaky_relu_(self._fc_style(style_emb), negative_slope=0.2)
        class_emb = F.leaky_relu_(self._fc_class(class_emb), negative_slope=0.2)
        # 2. Convolution
        x = torch.cat((style_emb, class_emb), dim=1)
        x = x.view(x.size(0), 128, 2, 2)
        x = self._deconv_blocks(x)
        # Return
        return x


# ----------------------------------------------------------------------------------------------------------------------
# Sprites
# ----------------------------------------------------------------------------------------------------------------------

class EncoderSprites(nn.Module):
    """
    Encoder Module.
    """
    def __init__(self, nz):
        super(EncoderSprites, self).__init__()
        # 1. Architecture
        # (1) Convolution
        self._conv_blocks = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=128, track_running_stats=True),
            nn.ReLU(inplace=True))
        # (2) FC
        self._fc = nn.Linear(in_features=512, out_features=nz, bias=True)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, x):
        x = self._conv_blocks(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        ret = self._fc(x)
        # Return
        return ret


class ReconstructorSprites(nn.Module):
    """
    Decoder Module.
    """
    def __init__(self, style_dim, class_dim, num_classes):
        super(ReconstructorSprites, self).__init__()
        # 1. Architecture
        self.register_parameter('word_dict', torch.nn.Parameter(torch.randn(size=(num_classes, class_dim))))
        # (1) FC
        self._fc_style = nn.Linear(in_features=style_dim, out_features=256, bias=True)
        self._fc_class = nn.Linear(in_features=class_dim, out_features=256, bias=True)
        # (2) Convolution
        self._deconv_blocks = nn.Sequential(
            # Layer 1
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=8, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 4
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid())
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, style_emb, class_label):
        # Get class dim
        class_emb = torch.index_select(self.word_dict, dim=0, index=class_label)
        # 1. FC
        style_emb = F.leaky_relu_(self._fc_style(style_emb), negative_slope=0.2)
        class_emb = F.leaky_relu_(self._fc_class(class_emb), negative_slope=0.2)
        # 2. Convolution
        x = torch.cat((style_emb, class_emb), dim=1)
        x = x.view(x.size(0), 128, 2, 2)
        x = self._deconv_blocks(x)
        # Return
        return x
