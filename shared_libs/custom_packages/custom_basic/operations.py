
import os
import copy
import torch
import shutil
import numpy as np
from collections import OrderedDict


########################################################################################################################
# Collectors & Containers
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Storage & Fetching
# ----------------------------------------------------------------------------------------------------------------------

class IterCollector(object):
    """
    Collecting items.
    """
    def __init__(self):
        self._dict = {}

    def __getitem__(self, key):
        return self._dict[key]

    @property
    def dict(self):
        return self._dict

    def _collect_method(self, value, self_value):
        if isinstance(value, dict):
            return {
                k: self._collect_method(v, self_value[k] if self_value is not None else None) for k, v in value.items()
            }
        else:
            if self_value is None:
                return [value]
            else:
                assert isinstance(self_value, list)
                self_value.append(value)
                return self_value

    def collect(self, items):
        if isinstance(items, ValidContainer): items = items.dict
        # Process each item
        for key, value in items.items():
            self_value = self._dict[key] if key in self._dict.keys() else None
            self._dict[key] = self._collect_method(value, self_value)

    def _pack_method(self, value, reduction):
        if isinstance(value, dict):
            return {k: self._pack_method(v, reduction) for k, v in value.items()}
        else:
            if value is None: return None
            # None value
            assert isinstance(value, list)
            # (1) Pack
            if isinstance(value[0], np.ndarray):
                value = np.concatenate(value, axis=0)
            elif isinstance(value[0], torch.Tensor):
                value = torch.cat(value, dim=0)
            else:
                value = np.array(value)
            # (2) Reduction
            if reduction == 'none':
                pass
            else:
                value = reduction(value)
            # Return
            return value

    def pack(self, reduction='none'):
        # Init result
        result = {}
        # Process each item
        for key, value in self._dict.items():
            result[key] = self._pack_method(value, reduction)
        # Return
        return result


class ValidContainer(object):
    """
    Container that doesn't tolerate None.
    """
    def __init__(self, *args, **kwargs):
        # Configs
        self._update_skip_none = chk_d(kwargs, 'update_skip_none')
        # Set dict
        container = {}
        for a in args:
            assert isinstance(a, dict) and len(set(a.keys()) - set(container.keys())) == len(a), "Duplicated keys. "
            container.update(a)
        self._dict = self._process(container)

    def __getitem__(self, key):
        return None if key not in self._dict.keys() else self._dict[key]

    def __setitem__(self, key, value):
        self.update({key: value})

    def _process(self, kwargs):
        # Init ret
        ret = OrderedDict()
        # Process
        for k, v in kwargs.items():
            if v is None: continue
            if isinstance(v, dict): v = self._process(v)
            ret[k] = v
        # Return
        return ret

    @property
    def dict(self):
        return self._dict

    def update(self, _dict, **kwargs):
        """
        :param _dict: 
        :param kwargs: 
            - skip_none: How to handle value that is None. 
        """
        if isinstance(_dict, ValidContainer): _dict = _dict.dict
        # Update
        for k, v in _dict.items():
            # Process value
            # (1) None
            if v is None:
                # Processing None value
                if k in self._dict.keys():
                    # Whether to skip
                    skip = self._update_skip_none
                    if 'skip_none' in kwargs.keys(): skip = kwargs['skip_none']
                    # Process
                    if not skip: self._dict.pop(k)
                # Move to next
                continue
            # (2) Dict
            if isinstance(v, dict): v = self._process(v)
            # Update
            self._dict[k] = v
        # Return
        return self


def fet_d(container, *args, **kwargs):
    """
    :param container:
    :param args:
    :param kwargs:
    :return:
        - policy_on_null: How to handle key that not exists.
        - pop: If pop out from container.

        - prefix: Prefix of keys to be fetched.
        - lambda processing keys:
            - remove    or
            - replace
        - lambda processing values:
            - lmd_v
    """
    if not isinstance(container, dict):
        container = container.dict
        assert 'pop' not in kwargs.keys()
    # ------------------------------------------------------------------------------------------------------------------
    # Preliminary
    # ------------------------------------------------------------------------------------------------------------------
    # 1. Policy
    policy_on_null = kwargs['policy_on_null'] if 'policy_on_null' in kwargs.keys() else 'skip'
    assert policy_on_null in ['ret_none', 'skip']
    # 2. Lambdas
    # (1) Processing key
    lmd_k = None
    # 1> Remove
    if 'remove' in kwargs.keys():
        tbr = kwargs['remove']
        # (1) Single str
        if isinstance(tbr, str):
            lmd_k = lambda _k: _k.replace(tbr, "")
        # (2) Multi str
        elif is_tuple_list(tbr):
            lmd_k_str = "lambda k: k"
            for t in tbr: lmd_k_str += ".replace('%s', '')" % t
            lmd_k = eval(lmd_k_str)
        # Others
        else:
            raise NotImplementedError
    # 2> Replace
    if 'replace' in kwargs.keys():
        assert lmd_k is None
        tbr = kwargs['replace']
        # (1) Prefix
        if isinstance(tbr, str):
            lmd_k = lambda _k: _k.replace(kwargs['prefix'], tbr)
        # Others
        else:
            raise NotImplementedError
    # (2) Processing values
    lmd_v = None if 'lmd_v' not in kwargs.keys() else kwargs['lmd_v']
    # ------------------------------------------------------------------------------------------------------------------
    # Fetching items
    # ------------------------------------------------------------------------------------------------------------------
    # 1. Collect keys
    if (not args) and 'prefix' not in kwargs.keys():
        args = list(container.keys())
    else:
        args = list(args)
        if 'prefix' in kwargs.keys(): args += list(filter(lambda k: k.startswith(kwargs['prefix']), container.keys()))
    # 2. Fetching items
    # (1) Init
    ret = OrderedDict()
    # (2) Process
    for k in args:
        # Fetching
        k_fetch, k_ret = k if is_tuple_list(k) else (k, k)
        if k not in container.keys():
            if policy_on_null == 'skip':
                continue
            else:
                v = None
        else:
            v = container[k_fetch] if not chk_d(kwargs, 'pop') else container.pop(k_fetch)
        # Processing key & value
        if lmd_k is not None: k_ret = lmd_k(k_ret)
        if lmd_v is not None: v = lmd_v(v)
        # Set
        ret[k_ret] = v
    # (3) Return
    return ret


# ----------------------------------------------------------------------------------------------------------------------
# Checking key
# ----------------------------------------------------------------------------------------------------------------------

def check_container(lmd_check_key, lmd_fetch_key, key, operator=None, another=None):
    # 1. Return False if key not exists
    if not lmd_check_key(key): return False
    # 2. Check
    if operator is None:
        return lmd_fetch_key(key)
    elif operator == 'not':
        return not lmd_fetch_key(key)
    elif callable(operator):
        if another is None:
            return operator(lmd_fetch_key(key))
        else:
            return operator(lmd_fetch_key(key), another)
    else:
        return eval('lmd_fetch_key(key) %s another' % operator)


def chk_d(container, key, operator=None, another=None):
    lmd_check_key = lambda k: k in container.keys()
    lmd_fetch_key = lambda k: container[k]
    return check_container(lmd_check_key, lmd_fetch_key, key, operator, another)


def chk_ns(container, key, operator=None, another=None):
    lmd_check_key = lambda k: hasattr(container, k)
    lmd_fetch_key = lambda k: getattr(container, k)
    return check_container(lmd_check_key, lmd_fetch_key, key, operator, another)


########################################################################################################################
# Others
########################################################################################################################

def is_tuple_list(val):
    return isinstance(val, tuple) or isinstance(val, list)


class TempDirManager(object):
    """
    Contextual manager. Create temporary directory for evaluations when entering, and delete when exiting.
    """
    def __init__(self, root_dir, *dir_names):
        self._root_dir = root_dir
        self._dir_names = dir_names
        self._dirs = []

    def __enter__(self):
        # Process each one
        for name in self._dir_names:
            temp_dir = os.path.join(self._root_dir, name)
            # Make directory
            assert not os.path.exists(temp_dir)
            os.makedirs(temp_dir)
            # Save
            self._dirs.append(temp_dir)
        # Return
        return self._dirs

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Delete temporary directories
        for dir_path in self._dirs:
            shutil.rmtree(dir_path)


class TempKwargsManager(object):
    """
    Config manager. Temporarily update config before an operation (when entering), and restore when exiting.
    """
    def __init__(self, instance, **kwargs):
        """
        :param instance: Should has attr _kwargs.
        :param kwargs: Configs to be updated.
        """
        # Members
        self._instance = instance
        self._kwargs = kwargs
        # Original kwargs
        self._orig_kwargs = getattr(self._instance, '_kwargs')

    def __enter__(self):
        if not self._kwargs: return
        # Get temporary cfg
        kwargs = copy.deepcopy(self._orig_kwargs)
        kwargs.update(self._kwargs)
        # Set
        setattr(self._instance, '_kwargs', kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._kwargs: return
        # Restore cfg
        setattr(self._instance, '_kwargs', self._orig_kwargs)
