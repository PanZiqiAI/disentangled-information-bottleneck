
import torch
from torch import nn
from shared_libs.utils.operations import gaussian_kl_div
from shared_libs.custom_packages.custom_pytorch.operations import BaseCriterion, TensorWrapper, wraps


class GaussKLDivLoss(BaseCriterion):
    """
    Gaussian KL divergence loss.
    """
    def __init__(self, model, lmd, **kwargs):

        # Applying convex hfunc
        @wraps(hyper_param=lmd)
        def _lambda_hfunc(_loss):
            if kwargs['hfunc'] == 'exp':
                return (_loss * kwargs['hfunc_param']).exp()
            elif kwargs['hfunc'] == 'pow':
                return _loss ** (1.0 + kwargs['hfunc_param'])
            elif kwargs['hfunc'] == 'none':
                return _loss
            else:
                raise NotImplementedError

        super(GaussKLDivLoss, self).__init__(lmd=_lambda_hfunc)
        # Config
        assert model in ['vib', 'nib']
        self._model = model

    def _call_method(self, params):
        if self._model == 'vib':
            return gaussian_kl_div(params1=params, average_batch=True)
        else:
            mu, std = params
            return gaussian_kl_div(params1=(mu.unsqueeze(1), std.unsqueeze(1)),
                                   params2=(mu.unsqueeze(0), std.unsqueeze(0))).mean()


class CrossEntropyLoss(BaseCriterion):
    """
    Classification loss.
    """
    def __init__(self, lmd=None):
        super(CrossEntropyLoss, self).__init__(lmd)
        # Config
        self._loss = nn.CrossEntropyLoss()

    def _call_method(self, output, label):
        return self._loss(output, label)


class RecLoss(BaseCriterion):
    """
    Reconstruction Loss.
    """
    def _call_method(self, ipt, target):
        loss_rec = torch.sum((ipt - target).pow(2)) / ipt.data.nelement()
        # Return
        return loss_rec


class EstLoss(BaseCriterion):
    """
    Estimator objective.
    """
    def __init__(self, radius, lmd=None):
        super(EstLoss, self).__init__(lmd=lmd)
        # Config
        self._radius = radius

    def _call_method(self, mode, **kwargs):
        assert mode in ['main', 'est']
        # 1. Calculate for main
        if mode == 'main':
            # (1) Density estimation
            loss_est = -kwargs['output'].mean()
            # (2) Making embedding located in [-radius, radius].
            emb = torch.cat(kwargs['emb'], dim=0)
            loss_wall = torch.relu(torch.abs(emb) - self._radius).square().mean()
            # Return
            return {'loss_est': loss_est, 'loss_wall': loss_wall}, -loss_est
        # 2. Calculate for estimator
        else:
            # (1) Real & fake losses
            loss_real = torch.mean((1.0 - kwargs['output_real']) ** 2)
            loss_fake = torch.mean((1.0 + kwargs['output_fake']) ** 2)
            # (2) Making outputs of the estimator to be zero-centric
            outputs = torch.cat([kwargs['output_real'], kwargs['output_fake']], dim=0)
            loss_zc = torch.mean(outputs).square()
            # Return
            return {'loss_real': loss_real, 'loss_fake': loss_fake, 'loss_zc': loss_zc}, \
                   (kwargs['output_real'].mean(), kwargs['output_fake'].mean())

    def __call__(self, mode, **kwargs):
        ret = super(EstLoss, self).__call__(mode, **kwargs)
        # 1. For main
        if mode == 'main':
            losses, est = ret if isinstance(ret, tuple) else (ret, TensorWrapper(None))
            losses.update({'est': est})
            # Return
            return losses
        # 2. For estimator
        else:
            losses, (est_real, est_fake) = ret if isinstance(ret, tuple) else (ret, TensorWrapper(None), TensorWrapper(None))
            losses.update({'est_real': est_real, 'est_fake': est_fake})
            # Return
            return losses


# ----------------------------------------------------------------------------------------------------------------------
# Discriminator
# ----------------------------------------------------------------------------------------------------------------------

class GANLoss(BaseCriterion):
    """
    GAN objectives.
    """
    def __init__(self, lmd=None):
        """
        Adversarial loss.
        """
        super(GANLoss, self).__init__(lmd)
        # Set loss
        self.__loss = nn.CrossEntropyLoss()

    def _call_method(self, pred, target_is_real):
        target_tensor = torch.tensor(1 if target_is_real else 0, dtype=torch.long).to(pred.device)
        loss = self.__loss(pred, target_tensor.expand(pred.size(0), ))
        # Return
        return loss, torch.max(pred, dim=1)[1]

    def __call__(self, prediction, target_is_real, **kwargs):
        # Get result
        ret = super(GANLoss, self).__call__(prediction, target_is_real, **kwargs)
        loss, pred = ret if isinstance(ret, tuple) else (ret, TensorWrapper(None))
        # Return
        return {'loss': loss, 'pred': pred}
