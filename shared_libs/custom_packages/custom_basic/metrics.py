
import time
import numpy as np
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from shared_libs.custom_packages.custom_basic.operations import chk_d


########################################################################################################################
# Meters
########################################################################################################################

class BestPerfMeter(object):
    """
    Meter to remember the best.
    """
    def __init__(self, early_stop_trials, iter_name, perf_name, lmd_ascend_perf=lambda new, stale: new > stale):
        # Configuration
        self._early_stop_trials = early_stop_trials
        self._iter_name, self._perf_name = iter_name, perf_name
        self._lmd_ascend_perf = lmd_ascend_perf
        # Data
        self._best_iter = None
        self._best_perf = None
        self._trials_no_ascend = 0
        
    @property
    def best_iter(self):
        return self._best_iter
    
    @property
    def best_perf(self):
        return self._best_perf

    @property
    def early_stop(self):
        if (self._early_stop_trials > 0) and (self._trials_no_ascend >= self._early_stop_trials):
            return True
        else:
            return False

    def set(self, val):
        self._best_iter = val['best_%s' % self._iter_name]
        self._best_perf = val['best_%s' % self._perf_name]

    def get(self):
        ret = OrderedDict()
        ret['best_%s' % self._iter_name] = self._best_iter
        ret['best_%s' % self._perf_name] = self._best_perf
        # Return
        return ret

    def update(self, iter_index, new_perf):
        # Better
        if self._best_perf is None or self._lmd_ascend_perf(new_perf, self._best_perf):
            # Ret current best iter as 'last_best_iter'
            ret = self._best_iter
            # Update
            self._best_iter = iter_index
            self._best_perf = new_perf
            self._trials_no_ascend = 0
            return ret
        # Update trials
        else:
            self._trials_no_ascend += 1
            return -1


class StopWatch(object):
    """
    Timer for recording durations.
    """
    def __init__(self):
        # Statistics - current
        self._stat = 'off'
        self._cur_duration = 0.0
        # Statistics - total
        self._total_duration = 0.0

    @property
    def stat(self):
        return self._stat

    def resume(self):
        # Record start time, switch to 'on'
        self._cur_duration = time.time()
        self._stat = 'on'

    def pause(self):
        if self._stat == 'off': return
        # Get current duration, switch to 'off'
        self._cur_duration = time.time() - self._cur_duration
        self._stat = 'off'
        # Update total duration
        self._total_duration += self._cur_duration

    def get_duration_and_reset(self):
        result = self._total_duration
        self._total_duration = 0.0
        return result


class FreqCounter(object):
    """
    Handling frequency.
    """
    def __init__(self, freq):
        assert freq > 0
        # Config
        self._freq = freq
        # Values
        self._count = 0
        self._status = False
        
    @property
    def status(self):
        return self._status

    def check(self, iteration, virtual=False):
        # Get count
        count = (iteration + 1) // self._freq
        # Update
        if count > self._count:
            if virtual: return True
            self._count = count
            self._status = True
        else:
            if virtual: return False
            self._status = False
        # Return
        return self._status


class TriggerLambda(object):
    """
    Triggered by a function.
    """
    def __init__(self, lmd_trigger):
        # Config
        self._lmd_trigger = lmd_trigger

    def check(self, n):
        return self._lmd_trigger(n)


class TriggerPeriod(object):
    """
    Trigger using period:
        For the example 'period=10, trigger=3', then 0,1,2 (valid), 3,4,5,6,7,8,9 (invalid).
        For the example 'period=10, trigger=-3', then 0,1,2,3,4,5,6 (invalid), 7,8,9 (valid).
    """
    def __init__(self, period, area):
        assert period > 0
        # Get lambda & init
        self._lmd_trigger = (lambda n: n < area) if area >= 0 else (lambda n: n >= period + area)
        # Configs
        self._period = period
        self._count = 0

    def check(self):
        # 1. Get return
        ret = self._lmd_trigger(self._count)
        # 2. Update counts
        self._count = (self._count + 1) % self._period
        # Return
        return ret


class _TimersManager(object):
    """
    Context manager for timers.
    """
    def __init__(self, timers, cache):
        # Config
        self._timers = timers
        self._cache = cache

    def __enter__(self):
        if self._cache is None: return
        # Activate
        for k in self._cache['on']:
            self._timers[k].resume()
        for k in self._cache['off']:
            self._timers[k].pause()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cache is None: return
        # Restore
        for k in self._cache['on']:
            self._timers[k].pause()
        for k in self._cache['off']:
            self._timers[k].resume()


class TimersController(object):
    """
    Controller for a bunch of timers.
    """
    def __init__(self, **kwargs):
        # Members
        self._timers = {}
        # Set timers
        for k, v in kwargs.items():
            self[k] = v

    def __contains__(self, key):
        return key in self._timers.keys()

    def __setitem__(self, key, val):
        assert key not in self._timers.keys() and isinstance(val, StopWatch) and val.stat == 'off'
        # Save
        self._timers[key] = val

    def __getitem__(self, key):
        return self._timers[key]

    def __call__(self, *args, **kwargs):
        # Calculate cache
        if not chk_d(kwargs, 'void'):
            # 1. On
            on = list(filter(lambda _k: _k in self._timers, args))
            for k in on: assert self._timers[k].stat == 'off'
            # 2. Off
            off = [k for k in filter(lambda _k: self._timers[_k].stat == 'on', self._timers.keys())]
            # Result
            cache = {'on': on, 'off': off}
        else:
            cache = None
        # Return
        return _TimersManager(self._timers, cache=cache)


# ----------------------------------------------------------------------------------------------------------------------
# Exponential Moving Average
# ----------------------------------------------------------------------------------------------------------------------

class EMA(object):
    """
    Exponential Moving Average.
    """
    def __init__(self, beta, init=None):
        super(EMA, self).__init__()
        # Config
        self._beta = beta
        # Set data
        self._stale = init

    @property
    def avg(self):
        return self._stale
    
    @avg.setter
    def avg(self, val):
        self._stale = val

    def update_average(self, new):
        # Update stale
        if new is not None:
            self._stale = new if self._stale is None else \
                self._beta * self._stale + (1.0 - self._beta) * new
        # Return
        return self._stale


class EMAPyTorchModel(object):
    """
    Exponential Moving Average for PyTorch Model.
    """
    def __init__(self, beta, model, **kwargs):
        # Config
        self._beta = beta
        # 1. Set data
        self._model, self._initialized = model, False
        # 2. Init
        if 'init' in kwargs.keys():
            self._model.load_state_dict(kwargs['init'].state_dict())
            self._initialized = True

    @property
    def initialized(self):
        return self._initialized

    @property
    def avg(self):
        return self._model

    @avg.setter
    def avg(self, val):
        self._model = val
    
    def update_average(self, new):
        # Update stale
        if new is not None:
            # 1. Init
            if not self._initialized: 
                self._model.load_state_dict(new.state_dict())
                self._initialized = True
            # 2. Moving average
            else:
                for stale_param, new_param in zip(self._model.parameters(), new.parameters()):
                    stale_param.data = self._beta * stale_param.data + (1.0 - self._beta) * new_param.data
        # Return
        return self._model


########################################################################################################################
# Metrics
########################################################################################################################

def conf_interval(data, axis=-1, conf_percent=90):
    """
    :param data: np.array.
    :param axis:
    :param conf_percent:
    :return:
    """
    # 1. Calculate avg & std.
    avg = np.mean(data, axis=axis, keepdims=True)
    std = np.sqrt(np.mean((data - avg)**2, axis=axis, keepdims=True))
    # 2. Calculate CI.
    # (1) Get 'n'
    n = {
        90: 1.645,
        95: 1.96,
        99: 2.576
    }[conf_percent]
    # (2) Get interval
    interval = n * std
    # Return
    return {'avg': np.squeeze(avg, axis=axis), 'std': np.squeeze(std, axis=axis), 'interval': np.squeeze(interval, axis=axis)}


def mean_accuracy(global_gt, global_pred, num_classes=None):
    """
    Mean Accuracy for classification.
    :param global_gt: (N, )
    :param global_pred: (N, )
    :param num_classes: Int. Provided for avoiding inference.
    :return:
    """
    # Infer num_classes
    if num_classes is None: num_classes = len(set(global_gt))
    # (1) Init result
    mean_acc = 0
    classes_acc = []
    # (2) Process each class
    for i in range(num_classes):
        cur_indices = np.where(global_gt == i)[0]
        # For current class
        cur_acc = accuracy_score(global_gt[cur_indices], global_pred[cur_indices])
        # Add
        mean_acc += cur_acc
        classes_acc.append(cur_acc)
    # (3) Get result
    mean_acc = mean_acc / num_classes
    # Return
    return mean_acc, np.array(classes_acc)
