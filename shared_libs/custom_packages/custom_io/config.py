
import os
import sys
import torch
import pickle
import random
import argparse
from argparse import Namespace
from shared_libs.custom_packages.custom_basic.operations import chk_d
from shared_libs.custom_packages.custom_io.logger import show_arguments


def str2bool(string):
    if string.lower() == 'true': return True
    elif string.lower() == 'false': return False
    else: raise ValueError


class CustomParser(object):
    """
    Argument generator.
    """
    def __init__(self, args_dict):
        """
        Param: args_dict used to record all args at current time.
        """
        self._args_dict = args_dict
        # Members
        self._parser, self._parser_names = argparse.ArgumentParser(allow_abbrev=False), []
        self._settings = {}

    def _check_duplicate(self, key):
        assert key not in self._parser_names, \
            "Key '%s' had already been added as a user-specified arguments. " % key
        assert key not in self._settings.keys(), \
            "Key '%s' has already been added as a determined setting with value '%s'. " % (key, self._settings[key])

    def add_argument(self, key, **kwargs):
        assert key.startswith("--"), "Argument key must start with '--'. "
        if key[2:] not in self._args_dict.keys():
            # Check duplicate
            self._check_duplicate(key[2:])
            # Set command
            if chk_d(kwargs, 'type', '==', bool):
                kwargs['type'] = str2bool
                if 'default' in kwargs.keys(): kwargs['default'] = str(kwargs['default'])
            self._parser.add_argument(key, **kwargs)
            # Save
            self._parser_names.append(key[2:])

    def set(self, key, value):
        if isinstance(key, list):
            for k, v in zip(key, value):
                self.set(k, v)
        else:
            if key not in self._args_dict.keys():
                self._check_duplicate(key)
                self._settings.update({key: value})

    def get_default(self, dest):
        return self._parser.get_default(dest)

    def get_args_dict(self):
        # Get respectively
        specified_args, _ = self._parser.parse_known_args()
        provided_args_dict = self._settings
        # Get result
        result = vars(specified_args)
        result.update(provided_args_dict)
        # Return
        return result, provided_args_dict.keys()


class TreeConfig(object):
    """
    1. Add root args;
    2. Add tree args according to current args:
        (1) Add arguments that are specified by user;
        (2) Add arguments that are uniquely determined;
    3. Add additional args;

    For convenience (typically loading config from file), arguments can be induced via args_inductor:
    1. Inducted args will be used during the entire building process of args tree;
    2. Arguments priorities:
            specifically provided
        =   determined (if their conditions have already been updated given inducted args)
        >   inducted args
        >   default args

    Arguments must be in the form of '--args=value'.
    """
    def __init__(self, args_inductor=None):
        # Parsing inducted_args from inductor
        self._inducted_args_dict = self._parse_inducted_args(args_inductor)
        # Get args & default args
        self._default_args_dict = {}
        ################################################################################################################
        # Stage 1: Add root args & parsing
        ################################################################################################################
        # 1. Add root args & parsing
        self.parser = CustomParser({})
        self._add_root_args()
        root_args_dict, provided_keys = self.parser.get_args_dict()
        # 2. Merge with inducted_args in terms of root_sys_args & save default
        self._collect_default_args(root_args_dict.keys())
        args_dict = self._merge_with_inducted_args(root_args_dict, provided_keys)
        ################################################################################################################
        # Stage 2: Add tree args & parsing via loop
        ################################################################################################################
        while True:
            # 1. Add tree args according to current args & parsing
            # (1) Reset
            self.parser = CustomParser(args_dict)
            # (2) Get args
            self._add_tree_args(args_dict)
            loop_args_dict, provided_keys = self.parser.get_args_dict()
            # 2. Get incremental args
            assert len(list(filter(lambda n: n in args_dict.keys(), loop_args_dict.keys()))) == 0
            # (2) Update or break
            if loop_args_dict:
                # 1> Merge with inducted_args for incremental_args & save default
                self._collect_default_args(loop_args_dict.keys())
                loop_args_dict = self._merge_with_inducted_args(loop_args_dict, provided_keys)
                # 2> Merge with stale
                args_dict = self._check_duplicated_args_and_merge(args_dict, loop_args_dict, 'stale', 'incremental')
            else:
                break
        ################################################################################################################
        # Stage 3: Add additional args & parsing
        ################################################################################################################
        # 1. Add additional args
        self.parser = CustomParser(args_dict)
        self._add_additional_args()
        additional_args_dict, provided_keys = self.parser.get_args_dict()
        # 2. Merge with inducted & save default
        self._collect_default_args(additional_args_dict.keys())
        additional_args_dict = self._merge_with_inducted_args(additional_args_dict, provided_keys)
        # 3. Update
        args_dict = self._check_duplicated_args_and_merge(args_dict, additional_args_dict, 'tree', 'additional')
        ################################################################################################################
        # Get final args
        ################################################################################################################
        # 1. Convert list to int, float ...
        # (1) Init result
        final_args_dict = {}
        # (2) Check each item
        for key, value in args_dict.items():
            # Try to convert
            convert = self._convert_list(value)
            # Save
            if convert is None:
                final_args_dict[key] = value
            else:
                final_args_dict[key] = convert
        # 2. Set
        self.args = Namespace(**final_args_dict)

        # Delete useless members
        del self.parser

    def _volatile_args(self):
        return []

    @staticmethod
    def _check_duplicated_args_and_merge(args_dict1, args_dict2, args_name1, args_name2):
        duplicated_names = list(filter(lambda name: name in args_dict1.keys(), args_dict2.keys()))
        assert len(duplicated_names) == 0, \
            'Duplicated args between %s and %s args: %s. ' % (args_name1, args_name2, str(duplicated_names))
        args_dict1.update(args_dict2)
        return args_dict1

    ####################################################################################################################
    # Args-convert
    ####################################################################################################################

    @staticmethod
    def _convert_list(value):
        # Must be str
        if not (isinstance(value, str) and value.startswith('[') and value.endswith(']')): return None
        # 1. Init result
        result = []
        # 2. Convert
        # (1) Split string
        split = value[1:-1].split(",")
        # (2) Check each split
        for data in split:
            # Empty
            if data == '': continue
            # Split
            if data.startswith('\'') and data.endswith('\''):
                result.append(data[1:-1])
            elif '.' in data:
                result.append(float(data))
            else:
                result.append(int(data))
        # Return
        return result

    ####################################################################################################################
    # Args-generate
    ####################################################################################################################

    def _parse_inducted_args(self, args_inductor):
        """
        This can be override, e.g., for loading from file to get induced_args.
        """
        return {}

    def _add_root_args(self):
        # Like: self.parser.add_argument(...)
        pass

    def _add_tree_args(self, args_dict):
        # Like: self.parser.add_argument(...) or
        #       self.parser.set(...)
        pass

    def _add_additional_args(self):
        # Like: self.parser.add_argument(...)
        pass

    @staticmethod
    def _collect_provided_sys_args_name():
        """
        Specifically provided and determined args have the highest priority.
        """
        # 1. Init result
        args_name = []
        # 2. Collect from commandline
        args_list = sys.argv[1:]
        for arg in args_list:
            assert str(arg).startswith("--"), "Arguments must be in the form of '--arg=value'. "
            arg_name = str(arg[2:]).split("=")[0]
            assert arg_name not in args_name, "Duplicated arg_name '%s'. " % arg_name
            args_name.append(arg_name)
        # Return
        return args_name

    def _collect_default_args(self, args_keys):
        for key in args_keys:
            assert key not in self._default_args_dict.keys()
            # Only specifically provided args can be collected to default dict
            if key in self._collect_provided_sys_args_name():
                self._default_args_dict.update({key: self.parser.get_default(key)})

    def _merge_with_inducted_args(self, args_dict, provided_keys):
        if not self._inducted_args_dict: return args_dict
        # 1. Collect highest priority args: provided & determined
        highest_priority_args = {
            key: args_dict[key]
            for key in filter(
                lambda name: name in args_dict.keys(), self._collect_provided_sys_args_name() + list(provided_keys))
        }
        # 2. Update inducted_args into sys_args
        args_dict.update({
            key: self._inducted_args_dict[key]
            for key in filter(lambda name: name in args_dict.keys(), self._inducted_args_dict.keys())
        })
        # 3. Override highest priority args
        args_dict.update(highest_priority_args)
        # Return
        return args_dict

    ####################################################################################################################
    # Utilization
    ####################################################################################################################

    def show_arguments(self, title=None, identifier='induced'):
        return show_arguments(
            self.args, title, self._volatile_args(),
            default_args=self._default_args_dict, compared_args=self._inducted_args_dict, competitor_name=identifier)

    def group_dict(self, prefix, replace=None):
        args = vars(self.args)
        # Select
        keys = list(filter(lambda k: k.startswith(prefix), args.keys()))
        # 1. Not replacing
        if replace is None:
            return {k: args[k] for k in keys}
        # 2. Replacing
        else:
            return {(replace + k[len(prefix):]): args[k] for k in keys}


class CanonicalConfig(TreeConfig):
    """
    Canonical Config.
    """
    def __init__(self, experiments_dir_path, load_rel_path=None):
        """
            Having the following built-in Arguments:
            (1) --desc:
            (2) --rand_seed
            (3) --load_from (for resuming training, typically an epoch or iteration index)
        and the following environmental arguments:
            (1) --exp_dir = given_exp_dir/given_desc/RandSeed[%d]
            (2) --params_dir = exp_dir/params
            (3) --analyses_dir = exp_dir/analyses
        """
        # Members
        self._exp_dir_path = experiments_dir_path
        # 1. Generate load_rel_path from given
        if load_rel_path is None:
            load_rel_path = self._generate_load_rel_path()
        # 2. If given, generating is not allowed
        else:
            assert self._generate_load_rel_path() is None
            # (1) Given is list of str.
            if not isinstance(load_rel_path, str):
                desc, rand_seed, load_from = load_rel_path
                load_rel_path = "%s/RandSeed[%s]/params/config[%s].pkl" % (desc, rand_seed, load_rel_path)
            # (2) Given is directly path
            else:
                pass
        # Super method for parsing configurations
        super(CanonicalConfig, self).__init__(args_inductor=load_rel_path)

        ################################################################################################################
        # Directories
        ################################################################################################################
        dirs = self._set_directory_args()
        # Make directories
        if self.args.load_from == -1:
            assert not os.path.exists(dirs[0]), "Rand seed '%d' has already been generated. " % self.args.rand_seed
            for d in dirs:
                if not os.path.exists(d): os.makedirs(d)

        # Delete useless members
        del self._exp_dir_path

    def _volatile_args(self):
        """
        These args will not be saved or displayed.
        """
        return ['desc'] + list(filter(lambda name: str(name).endswith('_dir'), vars(self.args).keys()))

    ####################################################################################################################
    # Args-generate
    ####################################################################################################################

    def _add_root_args(self):
        # (1) Description
        self.parser.add_argument("--desc",          type=str, default='unspecified')
        # (2) Rand Seed
        self.parser.add_argument("--rand_seed",     type=int, default=random.randint(0, 10000))
        # (3) Load_from
        self.parser.add_argument("--load_from",     type=int, default=-1)

    def _generate_load_rel_path(self):
        # 1. Init results
        load_args = {'desc': None, 'rand_seed': None, 'load_from': None}
        # 2. Check args
        for arg in sys.argv[1:]:
            assert str(arg).startswith("--"), "Arguments must be in the form of '--arg=value'. "
            if "=" in arg:
                assign_index = arg.find("=")
                arg_name, arg_value = arg[2:assign_index], arg[assign_index + 1:]
                if arg_name in load_args.keys(): load_args[arg_name] = arg_value
        # 3. Generate
        # Get rand_seed automatically
        none_args_name = list(filter(lambda key: load_args[key] is None, load_args.keys()))
        if len(none_args_name) == 1 and none_args_name[0] == 'rand_seed':
            desc_dir = os.path.join(self._exp_dir_path, load_args['desc'])
            trials_dirs = os.listdir(desc_dir)
            if len(trials_dirs) != 1:
                raise AssertionError("Too many trials in '%s'. Please specify a rand seed. " % desc_dir)
            else:
                load_args['rand_seed'] = int(str(trials_dirs[0]).split('[')[1].split(']')[0])
        # (1) Invalid
        for arg_value in load_args.values():
            if arg_value is None: return None
        # (2) Valid
        else:
            load_rel_path = "%s/RandSeed[%s]/params/config[%s].pkl" % (
                load_args['desc'], load_args['rand_seed'], load_args['load_from'])
            return load_rel_path

    def _parse_inducted_args(self, args_inductor):
        # Not loading
        if args_inductor is None:
            return {}
        # Loading
        else:
            ############################################################################################################
            # Get desc & rand_seed & load_from
            ############################################################################################################
            # 1. Get description
            params_dir, load_config_file = os.path.split(args_inductor)
            desc_with_rand_seed = os.path.split(params_dir)[0]
            inducted_args_desc, rand_seed = os.path.split(desc_with_rand_seed)
            # 2. Set random seed
            inducted_args_rand_seed = int(rand_seed.split("[")[1].split("]")[0])
            # 3. Set load_from
            inducted_args_load_from = int(load_config_file.split("[")[1].split("]")[0])
            ############################################################################################################
            # Load from config & update
            ############################################################################################################
            with open(os.path.join(self._exp_dir_path, args_inductor), 'rb') as f:
                saved_args_dict = pickle.load(f)
            saved_args_dict.update({
                'desc': inducted_args_desc,
                'rand_seed': inducted_args_rand_seed,
                'load_from': inducted_args_load_from
            })
            # Return
            return saved_args_dict

    def _set_directory_args(self, **kwargs):
        # Set directories
        self.args.desc = os.path.join(self.args.desc, 'RandSeed[%d]' % self.args.rand_seed)
        self.args.exp_dir = os.path.join(self._exp_dir_path, self.args.desc)
        self.args.params_dir = os.path.join(self.args.exp_dir, 'params')
        self.args.ana_dir = os.path.join(self.args.exp_dir, 'analyses')
        # Return
        return [self.args.exp_dir, self.args.params_dir, self.args.ana_dir]

    ####################################################################################################################
    # Utilization
    ####################################################################################################################

    def generate_save_path(self, n):
        return os.path.join(self.args.params_dir, 'config[%d].pkl' % n)

    def generate_checkpoint_path(self, n):
        return os.path.join(self.args.params_dir, 'checkpoint[%d].pth.tar' % n)

    def save(self, n, stale_n=None):
        save_path = self.generate_save_path(n)
        # 1. Save current
        if not os.path.exists(save_path):
            with open(save_path, 'wb') as f:
                args = vars(self.args)
                pickle.dump({k: args[k] for k in set(args.keys()) - set(self._volatile_args())}, f)
        # 2. Move stale
        if stale_n is not None:
            os.remove(self.generate_save_path(stale_n))

    def show_arguments(self, title=None, identifier='loaded'):
        return super(CanonicalConfig, self).show_arguments(title, identifier)


class CanonicalConfigTrainPyTorch(CanonicalConfig):
    """
    Canonical training config used in pytorch.
    """
    def __init__(self, exp_dir_path, load_rel_path=None, deploy_training_setting=True):
        super(CanonicalConfigTrainPyTorch, self).__init__(exp_dir_path, load_rel_path)
        # Init, typically checking compatibility
        self._init_method()
        # Deploy training settings
        if deploy_training_setting: self._deploy_training_settings()

    def _init_method(self):
        pass

    def _deploy_training_settings(self):
        """
        Set random_seed & deploy context.
        """
        # Random seed
        random.seed(self.args.rand_seed)
        torch.manual_seed(self.args.rand_seed)
        # Deploy context
        if self.args.gpu_ids != '-1':
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_ids
            self.args.device = 'cuda'
        else:
            self.args.device = 'cpu'

    def _set_directory_args(self, **kwargs):
        dirs = super(CanonicalConfigTrainPyTorch, self)._set_directory_args()
        # Tensorboard
        if not chk_d(kwargs, 'use_tfboard', 'not'):
            self.args.tfboard_dir = os.path.join(self._exp_dir_path, '../tensorboard')
            dirs.append(self.args.tfboard_dir)
        # Return
        return dirs

    def _add_root_args(self):
        super(CanonicalConfigTrainPyTorch, self)._add_root_args()
        # Context
        self.parser.add_argument("--gpu_ids",   type=str,   default='0', help="GPU ids. Set to -1 for CPU mode. ")
