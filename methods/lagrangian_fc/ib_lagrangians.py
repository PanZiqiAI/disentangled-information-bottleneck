
from shared_libs.modellib.fc import *
from shared_libs.utils.criterions import *
from shared_libs.utils.evaluations import *
from shared_libs.custom_packages.custom_basic.operations import fet_d, ValidContainer
from shared_libs.custom_packages.custom_basic.metrics import FreqCounter, BestPerfMeter
from shared_libs.custom_packages.custom_pytorch.base_models import EpochBatchBaseModel
from shared_libs.custom_packages.custom_io.logger import Logger, tfboard_add_multi_scalars
from shared_libs.custom_packages.custom_pytorch.operations import summarize_losses_and_backward


class IBLagrangianModel(EpochBatchBaseModel):
    """
    IB Lagrangian model.
    """
    def _build_architectures(self):
        # (1) Encoder
        if self._cfg.args.model == 'vib':
            encoder = VIBEncoder(self._cfg.args.input_dim, self._cfg.args.enc_hidden_dims, self._cfg.args.emb_dim,
                                 self._cfg.args.vib_softplus_scalar)
        elif self._cfg.args.model == 'nib':
            encoder = NIBEncoder(self._cfg.args.input_dim, self._cfg.args.enc_hidden_dims, self._cfg.args.emb_dim,
                                 self._cfg.args.nib_log_std, self._cfg.args.nib_log_std_trainable)
        else:
            raise NotImplementedError
        # (2) Decoder
        decoder = FCDecoder(self._cfg.args.emb_dim, self._cfg.args.dec_hidden_dims, self._cfg.args.num_classes)
        # Initialize
        super(IBLagrangianModel, self)._build_architectures(Enc=encoder, Dec=decoder)

    def _save_checkpoint(self, n, stale_n=None, **kwargs):
        super(IBLagrangianModel, self)._save_checkpoint(
            n, stale_n, items={'meter_best_perf': self._meters['meter_best_perf'].get()})

    def _load_checkpoint(self, **kwargs):
        super(IBLagrangianModel, self)._load_checkpoint(
            lmd_load_items=lambda _chkpt: self._meters['meter_best_perf'].set(_chkpt['meter_best_perf']))

    def _set_criterions(self):
        self._criterions['dec'] = CrossEntropyLoss(lmd=1.0)
        self._criterions['kl'] = GaussKLDivLoss(
            model=self._cfg.args.model, **self._cfg.group_dict("hfunc"), lmd=self._cfg.args.lambda_kl)

    def _set_optimizers(self):
        self._optimizers['default'] = torch.optim.Adam(self.parameters(), lr=self._cfg.args.learning_rate, betas=(0.5, 0.999))

    def _get_scheduler(self, optimizer, last_n):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.97 ** (epoch // 2))

    def _set_logs(self, **kwargs):
        super(IBLagrangianModel, self)._set_logs()

        def _get_logger(_name):
            return Logger(
                self._cfg.args.ana_dir, _name,
                formatted_prefix=self._cfg.args.desc, formatted_counters=['epoch', 'iter'],
                append_mode=False if self._cfg.args.load_from == -1 else True)

        # Evaluation
        self._logs['log_eval_acc'] = _get_logger('eval_accuracy')
        self._logs['log_eval_mi'] = _get_logger('eval_mi')
        self._logs['log_eval_attack'] = _get_logger('eval_attack_robustness')
        self._logs['log_eval_detection'] = _get_logger('eval_out_detection')

    def _set_meters(self, **kwargs):
        super(IBLagrangianModel, self)._set_meters()
        self._meters['meter_best_perf'] = BestPerfMeter(self._cfg.args.early_stop_trials, iter_name='epoch', perf_name='acc')

    def _deploy_batch_data(self, batch_data):
        x, label = map(lambda _x: _x.to(self._cfg.args.device), batch_data)
        return x.size(0), (x, label)

    def _train_batch(self, epoch, batch_index, iter_index, batch_data, packs):
        # Unpacking
        x, label = batch_data
        # --------------------------------------------------------------------------------------------------------------
        self._optimizers['default'].zero_grad()
        # 1. Forward calculation
        output = self._Dec(self._Enc(x))
        # 2. Calculate loss & backward
        loss_dec = self._criterions['dec'](output, label)
        loss_kl = self._criterions['kl'](self._Enc.params)
        # Backward & update
        summarize_losses_and_backward(loss_dec, loss_kl)
        self._optimizers['default'].step()
        # --------------------------------------------------------------------------------------------------------------
        """ Saving """
        packs['log'].update({
            'loss_dec': loss_dec.item(), 'loss_kl': loss_kl.item()
        })

    def _process_after_batch(self, epoch, batch_index, iter_index, packs, **kwargs):

        def _lmd_generate_log():
            r_tfboard = {
                'train/losses': fet_d(packs['log'], prefix='loss_')
            }
            packs['log'] = packs['log'].dict
            packs['tfboard'] = r_tfboard

        super(IBLagrangianModel, self)._process_after_batch(
            epoch, batch_index, iter_index, packs, lmd_generate_log=_lmd_generate_log)
        # Clear packs
        packs['log'] = ValidContainer()

    def _process_after_epoch(self, epoch, iter_index, packs):
        # 1. Update learning rate
        self._update_learning_rate()
        # 2. Evaluations
        # (1) Accuracy
        test_acc = self._api_eval_accuracy(epoch, iter_index, self._data['eval_train'], self._data['eval_test'])
        if self._meters['meter_best_perf'].update(epoch, test_acc) != -1: self._save_checkpoint(epoch)
        early_stop = self._meters['meter_best_perf'].early_stop
        # (2) Robustness, out detection & MI
        if early_stop or epoch == self._cfg.args.epochs - 1:
            # 1> Robustness
            self._api_eval_attack_robustness(epoch, iter_index, self._data['eval_train'], self._data['eval_test'])
            # 2> Out detection
            self._api_eval_out_detection(epoch, iter_index, self._data['eval_test'], self._cfg.args.eval_odin_out_data,
                                         out_name=self._cfg.args.eval_odin_out_data)
            # 3> Mutual information
            self._api_eval_mi(epoch, iter_index, self._data['eval_train'], self._data['eval_test'])
            # Saving params
            self._save_checkpoint(epoch)
        # Return
        return early_stop

    ####################################################################################################################
    # Evaluation
    ####################################################################################################################

    def _api_eval_accuracy(self, epoch, iter_index, train_dataloader, test_dataloader):
        evaluator = AccuracyEvaluator(self._Enc, self._Dec, device=self._cfg.args.device)
        # 1. Evaluating on train & test dataloader
        train_acc = evaluator(train_dataloader)
        test_acc = evaluator(test_dataloader)
        # 2. Logging
        self._logs['log_eval_acc'].info_formatted(
            counters={'epoch': epoch, 'iter': iter_index}, items={'train_acc': train_acc, 'test_acc': test_acc})
        tfboard_add_multi_scalars(
            self._logs['tfboard'], multi_scalars={
                'eval/acc': {'train': train_acc, 'test': test_acc}
            }, global_step=iter_index)
        # Return
        return test_acc

    def _api_eval_mi(self, epoch, iter_index, train_dataloader, test_dataloader):

        def _func_encode(_x):
            _emb = self._Enc(_x)
            return {'emb': _emb, 'params': self._Enc.params}

        evaluator = MIEvaluator(_func_encode, self._Dec, device=self._cfg.args.device)
        # 1. Evaluating on train & test dataloader
        ret = {}
        for d_name, dataloader in zip(['train', 'test'], [train_dataloader, test_dataloader]):
            ret['%s_mi_x_class' % d_name] = evaluator.eval_mi_x_z_monte_carlo(dataloader)
            ret['%s_mi_y_class' % d_name] = evaluator.eval_mi_y_z_variational_lb(dataloader)
        # 2. Logging
        self._logs['log_eval_mi'].info_formatted(counters={'epoch': epoch, 'iter': iter_index}, items=ret)
        tfboard_add_multi_scalars(
            self._logs['tfboard'], multi_scalars={
                'eval/mi_x_class': {d_name: ret['%s_mi_x_class' % d_name] for d_name in ['train', 'test']},
                'eval/mi_y_class': {d_name: ret['%s_mi_y_class' % d_name] for d_name in ['train', 'test']}
            }, global_step=iter_index)

    def _api_eval_attack_robustness(self, epoch, iter_index, train_dataloader, test_dataloader):
        torch.cuda.empty_cache()
        # 1. Evaluating
        evaluator = AdvAttackEvaluator(
            self._Enc, self._Dec, device=self._cfg.args.device, epsilon_list=self._cfg.args.eval_attack_epsilons)
        train_attack_acc = evaluator(train_dataloader)
        test_attack_acc = evaluator(test_dataloader)
        # 2. Logging
        items = {'train_%s' % k: v for k, v in train_attack_acc.items()}
        items.update({'test_%s' % k: v for k, v in test_attack_acc.items()})
        self._logs['log_eval_attack'].info_formatted(counters={'epoch': epoch, 'iter': iter_index}, items=items)
        tfboard_add_multi_scalars(
            self._logs['tfboard'], multi_scalars={
                'eval/attack_robustness': items
            }, global_step=iter_index)

    def _api_eval_out_detection(self, epoch, iter_index, test_dataloader, out_dataloader, out_name):
        torch.cuda.empty_cache()
        # 1. Evaluating
        evaluator = OutDetectionEvaluator(
            self._Enc, self._Dec, device=self._cfg.args.device,
            temper=self._cfg.args.eval_odin_temper, noise_magnitude=self._cfg.args.eval_odin_noise_mag,
            num_delta=self._cfg.args.eval_odin_num_delta)
        scores = evaluator(in_dataloader=test_dataloader, out_dataloader=out_dataloader)
        # 2. Logging
        self._logs['log_eval_detection'].info_formatted(
            counters={'epoch': epoch, 'iter': iter_index}, items={'%s_%s' % (out_name, k): v for k, v in scores.items()})
        tfboard_add_multi_scalars(
            self._logs['tfboard'], multi_scalars={
                'eval/out_detection_%s' % out_name: scores
            }, global_step=iter_index)
