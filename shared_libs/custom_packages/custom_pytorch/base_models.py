# Basic models that can be widely re-implemented.

import os
import torch
from torch import nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from shared_libs.custom_packages.custom_pytorch.operations import DataCycle
from shared_libs.custom_packages.custom_pytorch.operations import network_param_m
from shared_libs.custom_packages.custom_io.logger import Logger, tfboard_add_multi_scalars
from shared_libs.custom_packages.custom_basic.operations import ValidContainer, chk_d, chk_ns, fet_d
from shared_libs.custom_packages.custom_basic.metrics import FreqCounter, StopWatch, TimersController


class BaseModel(nn.Module):
    """
    Base model.
    ####################################################################################################################
    # Required cfg keys:
    ####################################################################################################################
        - tfboard_dir   (optional)

    ####################################################################################################################
    # Components
    ####################################################################################################################
        - self._logs['log_main']            (optional @ kwargs)
        - self._logs['tfboard']             (optional @ cfg.args.tfboard_dir & kwargs)

        - self._meters['timers']
            - 'io':                         (optional @ kwargs)
            - 'optimize':                   (optional @ kwargs)

    """
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        # Configurations
        self._cfg = cfg
        # Architectures
        self._build_architectures()
        
    def _build_architectures(self, **modules):
        """
        :param modules: { name1: module1, name2: module2 } 
        :return: Building modules. Skipping if module is None.
            self._networks = { name1: module1, name2: module2 }
            self._name1 = module1
            self._name2 = module2
        """
        assert modules
        # Dict to save all modules
        assert not hasattr(self, '_networks')
        self._networks = {}
        # Register modules
        for name, module in modules.items():
            if module is None: continue
            # Get attr name
            assert not name.startswith("_")
            # Save
            assert not hasattr(self, '_%s' % name)
            setattr(self, '_%s' % name, module)
            self._networks[name] = module

    ####################################################################################################################
    # Save & Load
    ####################################################################################################################

    def _selecting_modules_and_optimizers_for_chkpt(self, **kwargs):
        # 1. Modules
        if 'modules' not in kwargs.keys():
            modules = {'default': self}
        elif isinstance(kwargs['modules'], dict):
            modules = kwargs['modules']
        else:
            # Given should be list/tuple
            modules = {k: getattr(self, k) for k in kwargs['modules']}
        # 2. Optimizers
        if chk_ns(self, '_optimizers'):
            if 'optimizers' not in kwargs.keys():
                optimizers = self._optimizers
            elif kwargs['optimizers'] == 'sync_with_modules':
                optimizers = {k: self._optimizers[k] for k in modules.keys()}
            else:
                # Given should be list/tuple
                optimizers = {k: self._optimizers[k] for k in kwargs['optimizers']}
        else:
            optimizers = None
        # Return
        return modules, optimizers

    def _save_checkpoint(self, n, stale_n=None, **kwargs):
        # 1. Save checkpoint
        save_path = self._cfg.generate_checkpoint_path(n)
        if not os.path.exists(save_path):
            modules, optimizers = self._selecting_modules_and_optimizers_for_chkpt(**kwargs)
            # (1) Get final state to save (n, state_dict, optimizer)
            state = {
                'last': n,
                'state_dict': {k: v.state_dict() for k, v in modules.items()}
            }
            if optimizers is not None:
                state['optimizers'] = {k: v.state_dict() for k, v in optimizers.items()}
            # (2) Additional items
            if 'items' in kwargs.keys():
                for key, value in kwargs['items'].items():
                    assert key not in state.keys()
                    state[key] = value
            # Save
            torch.save(state, save_path)
        if stale_n is not None: os.remove(self._cfg.generate_checkpoint_path(stale_n))
        # 2. Save config
        self._cfg.save(n, stale_n)

    def _load_checkpoint(self, **kwargs):
        assert self._cfg.args.load_from != -1, "Please specify args.load_from. "
        modules, optimizers = self._selecting_modules_and_optimizers_for_chkpt(**kwargs)
        # 1. Load from file & check
        checkpoint_path = self._cfg.generate_checkpoint_path(self._cfg.args.load_from)
        checkpoint = torch.load(checkpoint_path)
        # 2. Restore
        assert checkpoint['last'] == self._cfg.args.load_from
        # (1) Parameters
        for k, module in modules.items():
            module.load_state_dict(checkpoint['state_dict'][k])
        # (2) Optimizer
        if optimizers is not None:
            for k, optimizer in optimizers.items():
                optimizer.load_state_dict(checkpoint['optimizers'][k])
        # (3) Additional items
        if 'lmd_load_items' in kwargs.keys():
            kwargs['lmd_load_items'](checkpoint)
        # Done
        print("Loaded from checkpoint '%s'. " % checkpoint_path)

    ####################################################################################################################
    # Train
    ####################################################################################################################

    # ------------------------------------------------------------------------------------------------------------------
    # Preliminaries - optimization
    # ------------------------------------------------------------------------------------------------------------------

    def _set_criterions(self):
        pass

    def _set_optimizers(self):
        pass

    def _get_scheduler(self, optimizer, last_n):
        pass

    def _set_schedulers(self):
        """
        Default: shared lr_scheduler for all optimizers.
        """
        for key, optimizer in self._optimizers.items():
            self._schedulers.update({key: self._get_scheduler(optimizer, self._cfg.args.load_from)})

    def _update_learning_rate(self):
        for scheduler in self._schedulers.values():
            if scheduler is not None: scheduler.step()

    def _set_logs(self, **kwargs):
        if not chk_d(kwargs, 'disable_log'):
            self._logs['log_main'] = Logger(
                self._cfg.args.ana_dir, 'train',
                formatted_prefix=self._cfg.args.desc, formatted_counters=kwargs['log_main_counters'],
                append_mode=False if self._cfg.args.load_from == -1 else True)
        if hasattr(self._cfg.args, 'tfboard_dir') and not chk_d(kwargs, 'disable_tfboard'):
            self._logs['tfboard'] = SummaryWriter(os.path.join(self._cfg.args.tfboard_dir, self._cfg.args.desc))

    def _set_meters(self, **kwargs):
        # Timers
        self._meters['timers'] = TimersController()
        if not chk_d(kwargs, 'disable_timers'):
            self._meters['timers']['io'] = StopWatch()
            self._meters['timers']['opt'] = StopWatch()

    def _deploy(self):
        # GPU
        if self._cfg.args.device == 'cuda':
            self.cuda()

    def _set_to_train_mode(self):
        self.train()

    # ------------------------------------------------------------------------------------------------------------------
    # APIs
    # ------------------------------------------------------------------------------------------------------------------

    def train_parameters(self, **kwargs):
        # Setup
        self._setup(**kwargs)
        # Logging before training
        self._logging()
        # Train parameters        
        self._train_procedure()

    def _setup(self, **kwargs):

        def _check_and_set_dict(name, value=None):
            assert not hasattr(self, name)
            if value is None:
                setattr(self, name, {})
                getattr(self, '_set%s' % name)()
            else:
                setattr(self, name, value)

        # 1. Setup data
        _check_and_set_dict('_data', {
            key[:-len('_data')]: kwargs[key]
            for key in filter(lambda k: k.endswith("_data") and kwargs[k] is not None, kwargs.keys())
        })
        # 2. Prepare for training
        # (1) Set for optimization
        _check_and_set_dict('_logs')
        _check_and_set_dict('_optimizers')
        _check_and_set_dict('_schedulers')
        _check_and_set_dict('_criterions')
        _check_and_set_dict('_meters')
        # (2) Resume
        if self._cfg.args.load_from != -1:
            self._load_checkpoint()
        # (3) Deploy network to GPUs.
        self._deploy()

    def _logging(self, **kwargs):
        """
        Logging before training.
        :return: 
        """
        for key, logger in self._logs.items():
            if not isinstance(logger, Logger): continue
            to_screen = key == ('log_main' if 'log_main' not in kwargs.keys() else kwargs['log_main'])
            # Show information
            logger.info_individually(self._cfg.show_arguments(), to_screen=to_screen)
            # (1) Show dataset
            if 'len_data' not in kwargs.keys():
                len_data = {}
                for n, dl in self._data.items():
                    if isinstance(dl, DataLoader):
                        len_data[n] = len(dl.dataset)
                    elif isinstance(dl, DataCycle):
                        len_data[n] = dl.num_samples
                    else:
                        raise NotImplementedError
            else:
                len_data = kwargs['len_data']
            for n, l in len_data.items():
                logger.info_individually("Dataset[%s] size: %d. " % (n, l), to_screen=to_screen)
            # (2) Show network
            for net_name, network in self._networks.items():
                logger.info_individually(
                    "Network[%s] total number of parameters : %.3f M. " % (net_name, network_param_m(network)), to_screen)

    def _train_procedure(self):
        raise NotImplementedError

    def _init_packs(self, *args, **kwargs):
        """ Init packages. """
        def _init(_k):
            return ValidContainer(**(kwargs[_k] if _k in kwargs.keys() else {}))
        # 1. Init
        ret = {}
        # 2. Set packages
        # (1) Log
        if 'log_main' in self._logs.keys() and not chk_d(kwargs, 'disable_log'):
            assert 'log' not in args
            ret['log'] = _init('log')
        # (2) TFBoard
        if 'tfboard' in self._logs.keys() and not chk_d(kwargs, 'disable_tfboard'):
            assert 'tfboard' not in args
            ret['tfboard'] = _init('tfboard')
        # (3) Others
        if len(args) > 0:
            assert len(set(args)) == len(args)
            for k in args: ret[k] = _init(k)
        # Return
        return ret

    def _deploy_batch_data(self, batch_data):
        raise NotImplementedError


class EpochBatchBaseModel(BaseModel):
    """
    Epoch-batch based training model.
    ####################################################################################################################
    # Required cfg keys:
    ####################################################################################################################
        - epochs
        - freq_iter_log         (optional)
        - freq_epoch_chkpt      (optional)
        - (base) tfboard_dir    (optional)

    ####################################################################################################################
    # Components
    ####################################################################################################################
        - (base) self._logs['log_main']             (optional @ kwargs)
        - (base) self._logs['tfboard']              (optional @ cfg.args.tfboard_dir & kwargs)

        - (base) self._meters['timers']
            - 'io':                                 (optional @ kwargs)
            - 'optimize':                           (optional @ kwargs)

        - self._meters['counter_log']               (optional @ cfg.args.freq_iter_log)
        - self._meters['counter_chkpt']             (optional @ cfg.args.freq_epoch_chkpt)

    """
    ####################################################################################################################
    # Train
    ####################################################################################################################

    # ------------------------------------------------------------------------------------------------------------------
    # Preliminaries - optimization
    # ------------------------------------------------------------------------------------------------------------------

    def _set_logs(self, **kwargs):
        log_main_counters = ['epoch', 'batch', 'iter'] if 'log_main_counters' not in kwargs.keys() else \
            kwargs.pop('log_main_counters')
        super(EpochBatchBaseModel, self)._set_logs(log_main_counters=log_main_counters, **kwargs)

    def _set_meters(self, **kwargs):
        super(EpochBatchBaseModel, self)._set_meters(**kwargs)
        # Counters
        if chk_ns(self._cfg.args, 'freq_iter_log', '>', 0):
            self._meters['counter_log'] = FreqCounter(self._cfg.args.freq_iter_log)
        if chk_ns(self._cfg.args, 'freq_epoch_chkpt', '>', 0):
            self._meters['counter_chkpt'] = FreqCounter(self._cfg.args.freq_epoch_chkpt)

    # ------------------------------------------------------------------------------------------------------------------
    # APIs
    # ------------------------------------------------------------------------------------------------------------------

    def _train_procedure(self, **kwargs):
        """
        Training procedure. 
        """
        # 1. Initialize packs used in training
        packs = self._init_packs()
        # 2. Training
        iter_index = -1
        for epoch in range(self._cfg.args.load_from + 1, self._cfg.args.epochs):
            # 1. Train each batch
            # Start recording io time
            if 'io' in self._meters['timers']: self._meters['timers']['io'].resume()
            # Read batch data
            for batch_index, batch_data in enumerate(self._data['train']):
                # Deploy
                batch_iters, batch_data = self._deploy_batch_data(batch_data)
                iter_index += batch_iters
                # End recording io time & start recording optimize time
                if 'io' in self._meters['timers']: self._meters['timers']['io'].pause()
                # Batch optimization
                with self._meters['timers']('opt', void=chk_d(kwargs, 'dis_t_opt')):
                    self._set_to_train_mode()
                    self._train_batch(epoch, batch_index, iter_index, batch_data, packs)
                ########################################################################################################
                # After-batch operations
                ########################################################################################################
                self._process_after_batch(epoch, batch_index, iter_index, packs)
            # 2. Process after epoch
            early_stop = self._process_after_epoch(epoch, iter_index, packs)
            if early_stop: return
        # Save final results
        self._save_checkpoint(self._cfg.args.epochs - 1)

    def _train_batch(self, epoch, batch_index, iter_index, batch_data, packs):
        raise NotImplementedError

    def _process_after_batch(self, epoch, batch_index, iter_index, packs, **kwargs):
        # Logging
        if chk_d(self._meters, 'counter_log', lambda c: c.check(iter_index)):
            if 'lmd_generate_log' in kwargs.keys(): kwargs['lmd_generate_log']()
            # (1) Logs
            if 'log_main' in self._logs.keys() and not chk_d(kwargs, 'disable_log'):
                # Update io & optimize timers
                if 'io' in self._meters['timers']:
                    packs['log']['t_io'] = self._meters['timers']['io'].get_duration_and_reset()
                if 'opt' in self._meters['timers']:
                    packs['log']['t_opt'] = self._meters['timers']['opt'].get_duration_and_reset()
                # Show information
                log_kwargs = {'items': packs['log']} if 'lmd_process_log' not in kwargs.keys() else \
                    kwargs['lmd_process_log'](packs['log'])
                self._logs['log_main'].info_formatted([epoch, batch_index, iter_index], **log_kwargs)
            # (2) Tensorboard
            if 'tfboard' in self._logs.keys() and not chk_d(kwargs, 'disable_tfboard'):
                tfboard_add_multi_scalars(self._logs['tfboard'], packs['tfboard'], global_step=iter_index)

    def _process_after_epoch(self, epoch, iter_index, packs):
        """
        :rtype: Whether to early stop training (bool), none by default.
        """
        # Update learning rate
        self._update_learning_rate()
        # Save current epoch
        if chk_d(self._meters, 'counter_chkpt', lambda c: c.check(epoch)):
            self._save_checkpoint(epoch)


class IterativeBaseModel(BaseModel):
    """
    Iteratively training model.
    ####################################################################################################################
    # Required cfg keys:
    ####################################################################################################################
        - steps or iters
        - freq_iter_log         (optional)
        - freq_step_chkpt       (optional)
        - (base) tfboard_dir

    ####################################################################################################################
    # Components
    ####################################################################################################################
        - (base) self._logs['log_main']             (optional @ kwargs)
        - (base) self._logs['tfboard']              (optional @ cfg.args.tfboard_dir & kwargs)

        - (base) self._meters['timers']
            - 'io':                                 (optional @ kwargs)
            - 'optimize':                           (optional @ kwargs)

        - self._meters['i']
        - self._meters['counter_log']               (optional @ cfg.args.freq_iter_log)
        - self._meters['counter_chkpt']             (optional @ cfg.args.freq_step_chkpt)

    """
    ####################################################################################################################
    # Save & Load
    ####################################################################################################################

    def _save_checkpoint(self, n, stale_n=None, **kwargs):
        # Save iterations
        items = {'i': self._meters['i']}
        # Merge with given
        if 'items' in kwargs.keys():
            items_given = kwargs.pop('items')
            assert 'i' not in set(items_given.keys())
            items.update(items_given)
        # Set
        super(IterativeBaseModel, self)._save_checkpoint(n, stale_n, items=items, **kwargs)

    def _load_checkpoint(self, **kwargs):
        # Load iterations
        def lmd_load_iterations(chkpt):
            self._meters['i'] = chkpt['i']
        lmd_ = lmd_load_iterations
        # Merge with given
        if 'lmd_load_items' in kwargs.keys():
            lmd_given = kwargs.pop('lmd_load_items')

            def lmd_load_items(chkpt):
                lmd_load_iterations(chkpt)
                lmd_given(chkpt)
            lmd_ = lmd_load_items
        # Set
        super(IterativeBaseModel, self)._load_checkpoint(lmd_load_items=lmd_, **kwargs)

    ####################################################################################################################
    # Train
    ####################################################################################################################

    # ------------------------------------------------------------------------------------------------------------------
    # Preliminaries - optimization
    # ------------------------------------------------------------------------------------------------------------------

    def _set_logs(self, **kwargs):
        # Get counters for main log
        if 'log_main_counters' in kwargs.keys():
            log_main_counters = kwargs.pop('log_main_counters')
        else:
            log_main_counters = ['epoch', 'batch', 'step', 'iter']
        # Set logs
        super(IterativeBaseModel, self)._set_logs(log_main_counters=log_main_counters, **kwargs)

    def _set_meters(self, **kwargs):
        super(IterativeBaseModel, self)._set_meters(**kwargs)
        # Counters
        if chk_ns(self._cfg.args, 'freq_iter_log', '>', 0):
            self._meters['counter_log'] = FreqCounter(self._cfg.args.freq_iter_log)
        if chk_ns(self._cfg.args, 'freq_step_chkpt', '>', 0):
            self._meters['counter_chkpt'] = FreqCounter(self._cfg.args.freq_step_chkpt)

    # ------------------------------------------------------------------------------------------------------------------
    # APIs
    # ------------------------------------------------------------------------------------------------------------------

    def _train_procedure(self, **kwargs):
        """
        Training procedure
        """
        # 1. Preliminaries
        iter_marker, iter_max = self._set_iterations()
        packs = self._init_packs()
        # 2. Main
        while self._meters['i'][iter_marker] < iter_max:
            # 1. Train
            with self._meters['timers']('opt', void=chk_d(kwargs, 'dis_t_opt')):
                self._set_to_train_mode()
                self._train_step(packs)
            # 2. Process after train
            early_stop = self._process_after_step(packs)
            if early_stop: return
            # Move forward
            self._meters['i']['step'] += 1
        # Save final result
        self._save_checkpoint(self._meters['i'][iter_marker] - 1)

    def _set_iterations(self, **kwargs):
        # Init iterations
        self._meters['i'] = {}
        # (1) Involving dataset
        if not chk_d(kwargs, 'disable_epoch_batch'):
            self._meters['i']['num_train_samples'] = self._data['train'].num_samples \
                if 'num_train_samples' not in kwargs.keys() else kwargs.pop('num_train_samples')
            self._meters['i'].update({'epoch': 0, 'batch': -1, 'num_cur_epoch': 0})
        # (2) Step & iter
        self._meters['i'].update({'step': 0, 'iter': -1})
        # Generate
        if hasattr(self._cfg.args, 'steps'):
            assert not hasattr(self._cfg.args, 'iters')
            return 'step', self._cfg.args.steps
        elif hasattr(self._cfg.args, 'iters'):
            assert not hasattr(self._cfg.args, 'steps')
            return 'iter', self._cfg.args.iters
        else: raise ValueError

    def _fetch_batch_data(self, **kwargs):
        record = not chk_d(kwargs, 'no_record')
        # Fetch data & update iterations
        with self._meters['timers']('io'):
            # Fetch data
            batch_iters, batch_data_deployed = self._deploy_batch_data(next(self._data['train']))
            # Update iterations
            if record:
                # (1) Update iter_index
                self._meters['i']['iter'] += batch_iters
                # (2) Move forward
                if 'epoch' in self._meters['i'].keys():
                    num_cur_epoch = self._meters['i']['num_cur_epoch'] + batch_iters
                    num_train_samples = self._meters['i']['num_train_samples']
                    if num_cur_epoch >= num_train_samples:
                        self._meters['i']['num_cur_epoch'] = num_cur_epoch % num_train_samples
                        self._meters['i']['batch'] = 0
                        self._meters['i']['epoch'] += 1
                    else:
                        self._meters['i']['num_cur_epoch'] = num_cur_epoch
                        self._meters['i']['batch'] += 1
        # Return
        return batch_data_deployed

    def _train_step(self, packs):
        raise NotImplementedError

    def _process_after_step(self, packs, **kwargs):
        """
        :rtype: Whether to early stop. None by default.
        """
        self._process_log_after_step(packs)
        self._process_chkpt_and_lr_after_step()

    def _process_log_after_step(self, packs, **kwargs):
        # Get iteration for log & tfboard
        iter_checker = kwargs['iter_checker'] if 'iter_checker' in kwargs.keys() else 'iter'
        # Logging
        if chk_d(self._meters, 'counter_log', lambda c: c.check(self._meters['i'][iter_checker])):
            if 'lmd_generate_log' in kwargs.keys(): kwargs['lmd_generate_log']()
            # (1) Logs
            if 'log_main' in self._logs.keys() and 'log' in packs.keys() and not chk_d(kwargs, 'disable_log'):
                # Update io & optimize timers
                if 'io' in self._meters['timers']:
                    packs['log']['t_io'] = self._meters['timers']['io'].get_duration_and_reset()
                if 'opt' in self._meters['timers']:
                    packs['log']['t_opt'] = self._meters['timers']['opt'].get_duration_and_reset()
                # Show information
                log_kwargs = {'items': packs['log']} if 'lmd_process_log' not in kwargs.keys() else \
                    kwargs['lmd_process_log'](packs['log'])
                self._logs['log_main'].info_formatted(
                    fet_d(self._meters['i'], *self._logs['log_main'].formatted_counters), **log_kwargs)
            # (2) Tensorboard
            if 'tfboard' in self._logs.keys() and 'tfboard' in packs.keys() and not chk_d(kwargs, 'disable_tfboard'):
                tfboard_add_multi_scalars(self._logs['tfboard'], packs['tfboard'], self._meters['i'][iter_checker])

    def _process_chkpt_and_lr_after_step(self, **kwargs):
        # Get iteration for chkpt
        iter_checker = kwargs['iter_checker'] if 'iter_checker' in kwargs.keys() else 'step'
        # 1. Learning rate
        self._update_learning_rate()
        # 2. Chkpt
        if chk_d(self._meters, 'counter_chkpt', lambda c: c.check(self._meters['i'][iter_checker])):
            self._save_checkpoint(n=self._meters['i'][iter_checker])
