
import torch
from torch import nn
from torch.nn import init
from shared_libs.custom_packages.custom_basic.operations import chk_ns


########################################################################################################################
# Data
########################################################################################################################

class DataCycle(object):
    """
    Data cycle infinitely. Using next(self) to fetch batch data.
    """
    def __init__(self, dataloader):
        # Dataloader
        self._dataloader = dataloader
        # Iterator
        self._data_iterator = iter(self._dataloader)

    @property
    def num_samples(self):
        return len(self._dataloader.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._data_iterator)
        except StopIteration:
            self._data_iterator = iter(self._dataloader)
            return next(self._data_iterator)


########################################################################################################################
# Networks
########################################################################################################################

def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)   -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # Weight
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            # Bias
            if chk_ns(m, 'bias', 'is not', None):
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def fix_grad(net):
    """
    Fix gradient.
    :param net:
    :return:
    """
    if isinstance(net, list):
        for layer in net:
            for p in layer.parameters():
                p.requires_grad = False
    else:
        for p in net.parameters():
            p.requires_grad = False


def set_requires_grad(nets, requires_grad=False):
    """
    Set requires_grad=False for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def set_requires_grad_and_mode(nets, requires_grad, mode):
    # Check
    assert mode in ['train', 'eval']
    if not isinstance(nets, list): nets = [nets]
    # Set gradients
    set_requires_grad(nets, requires_grad)
    # Set mode
    for net in nets:
        if net is not None:
            net.train() if mode == 'train' else net.eval()


def network_param_m(network):
    return sum([param.numel() for param in network.parameters()]) / 1e6


########################################################################################################################
# Criterion
########################################################################################################################

# ======================================================================================================================
# Typical usage:
#    Set lambda to positive if you want to both check(show in log) & back_prop.
#    Set lambda to 0 if you want to totally disable the loss, \ie, neither check nor back_prop.
#    Set lambda to negative (-1) if you want to only check (show in log).
# ----------------------------------------------------------------------------------------------------------------------

class TensorWrapper(object):
    """
    Tensor wrapper.
    """
    def __init__(self, tensor):
        self._tensor = tensor

    def item(self):
        return None if self._tensor is None else self._tensor.item()


class LossWrapper(TensorWrapper):
    """
    Loss wrapper.
    """
    def __init__(self, _lmd, loss_tensor):
        super(LossWrapper, self).__init__(loss_tensor)
        # Lambda
        self._lmd = _lmd

    def loss_backprop(self):
        if self._lmd.hyper_param > 0.0 and self._tensor is not None:
            return self._lmd(self._tensor) * self._lmd.hyper_param
        else:
            return None


def summarize_losses_and_backward(*args, **kwargs):
    """
    Each arg should either be instance of
        - None
        - Tensor
        - LossWrapper
        - LossWrapperContainer
    """
    # 1. Init
    ret = 0.0
    # 2. Summarize to result
    for arg in args:
        if arg is None:
            continue
        elif isinstance(arg, LossWrapper):
            loss_backprop = arg.loss_backprop()
            if loss_backprop is not None: ret += loss_backprop
        elif isinstance(arg, torch.Tensor):
            ret += arg
        else:
            raise NotImplementedError
    # 3. Backward
    if isinstance(ret, torch.Tensor):
        ret.backward(**kwargs)


def wraps(hyper_param):

    def update_lmd(_lmd):
        setattr(_lmd, 'hyper_param', hyper_param)
        return _lmd

    return update_lmd


class BaseCriterion(object):
    """
    Base criterion class.
    """
    def __init__(self, lmd=None):
        """
        Dynamic lambda if given is None.
        """
        # Config
        self._lmd = self._get_lmd(lmd) if lmd is not None else None

    @staticmethod
    def _get_lmd(_lmd):
        """
        _lmd:
            float: Will be reformed to a function with attribute "hyper_param".
            Function: If has not attribute "hyper_param", the attr will be set.
        Return: A function that maps original_tensor (produced by _call_method) to the tensor for backward (without
        multiplied the hyper_param).
        """
        def __get_single_lmd(_l):
            if isinstance(_l, float):
                _l = wraps(hyper_param=_l)(lambda x: x)
            else:
                if not hasattr(_l, "hyper_param"):
                    _l = wraps(hyper_param=1.0)(_l)
            # Return
            return _l

        # Single
        if not isinstance(_lmd, dict):
            return __get_single_lmd(_lmd)
        else:
            return {k: __get_single_lmd(_lmd[k]) for k in _lmd.keys()}

    def _call_method(self, *args, **kwargs):
        """
        For lambda is a number, return the corresponding loss tensor.
        For lambda is a dict, return a loss tensor dict that corresponds the lambda dict.
        """
        raise NotImplementedError

    @staticmethod
    def _get_loss_wrappers(_lmd, loss_tensor=None):
        # 1. For single lambda
        if not isinstance(_lmd, dict):
            # (1) Shared lambda
            if isinstance(loss_tensor, dict):
                return {key: LossWrapper(_lmd, value) for key, value in loss_tensor.items()}
            # (2) Single tensor.
            else:
                return LossWrapper(_lmd, loss_tensor)
        # 2. For multi lambda
        else:
            return {key: LossWrapper(_lmd[key], loss_tensor[key] if loss_tensor is not None else None)
                    for key in _lmd.keys()}

    @staticmethod
    def _need_calculate(_lmd):
        # 1. Lmd is a scalar
        if not isinstance(_lmd, dict):
            if _lmd.hyper_param == 0.0:
                return False
            else:
                return True
        # 2. Multi lmd
        else:
            for _lmd in _lmd.values():
                if _lmd.hyper_param != 0.0:
                    return True
            return False

    def __call__(self, *args, **kwargs):
        # Get lambda & Check if not calculate
        if 'lmd' in kwargs.keys():
            _lmd = self._get_lmd(kwargs.pop('lmd'))
        else:
            assert self._lmd is not None
            _lmd = self._lmd
        if not self._need_calculate(_lmd): return self._get_loss_wrappers(_lmd)
        # 1. Calculate
        call_result = self._call_method(*args, **kwargs)
        # 2. Return
        if isinstance(call_result, tuple):
            loss_tensor, others = call_result
            return self._get_loss_wrappers(_lmd, loss_tensor), others
        else:
            return self._get_loss_wrappers(_lmd, call_result)

# ======================================================================================================================


########################################################################################################################
# Others
########################################################################################################################

def collect_weight_keys(module, prefix=None, destination=None):
    """
    Collecting a module's weight (linear & conv).
    """
    # Check compatibility
    if destination is None:
        assert prefix is None
        destination = []
    else:
        assert prefix is not None
    # 1. For module is an instance of nn.Linear or nn.Conv
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        # Get weight
        state_dict = module.state_dict()
        weight_key = list(filter(lambda k: k.endswith("weight"), state_dict.keys()))
        assert len(weight_key) == 1
        weight_key = weight_key[0]
        # Save to result & return
        destination.append(("%s.%s" % (prefix, weight_key)) if prefix is not None else weight_key)
        return destination
    # 2. Recursive
    else:
        for key, sub_module in module._modules.items():
            destination = collect_weight_keys(
                sub_module, prefix=("%s.%s" % (prefix, key)) if prefix is not None else key,
                destination=destination)
        return destination


def sampling_z(batch_size, nz, device, random_type, **kwargs):
    if random_type == 'uni':
        # [-1, 1]
        z = torch.rand(batch_size, nz).to(device) * 2.0 - 1.0
        # [-radius, radius]
        if 'random_uni_radius' in kwargs.keys():
            z = z * kwargs['random_uni_radius']
        # [-radius + c, radius + c]
        if 'random_uni_center' in kwargs.keys():
            z = z + kwargs['random_uni_center']
    elif random_type == 'gauss':
        z = torch.randn(batch_size, nz).to(device)
    else:
        raise NotImplementedError
    return z

