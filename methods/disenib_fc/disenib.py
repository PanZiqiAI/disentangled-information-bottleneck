
from functools import partial
from shared_libs.modellib.fc import *
from shared_libs.utils.criterions import *
from shared_libs.utils.operations import *
from shared_libs.utils.evaluations import *
from shared_libs.custom_packages.custom_basic.operations import fet_d, ValidContainer
from shared_libs.custom_packages.custom_pytorch.base_models import IterativeBaseModel
from shared_libs.custom_packages.custom_io.logger import Logger, tfboard_add_multi_scalars
from shared_libs.custom_packages.custom_basic.metrics import FreqCounter, TriggerLambda, TriggerPeriod
from shared_libs.custom_packages.custom_pytorch.operations import summarize_losses_and_backward, set_requires_grad


class DisenIB(IterativeBaseModel):
    """
    Disentangled IB model (FC).
    """
    def _build_architectures(self):
        super(DisenIB, self)._build_architectures(
            # Encoder, decoder, reconstructor, estimator
            Enc_style=DisenIBEncoder(self._cfg.args.input_dim, self._cfg.args.enc_hidden_dims, self._cfg.args.style_dim),
            Enc_class=DisenIBEncoder(self._cfg.args.input_dim, self._cfg.args.enc_hidden_dims, self._cfg.args.class_dim),
            Dec=FCDecoder(self._cfg.args.class_dim, self._cfg.args.dec_hidden_dims, self._cfg.args.num_classes),
            Rec=FCReconstructor(self._cfg.args.style_dim, self._cfg.args.class_dim, self._cfg.args.rec_hidden_dims,
                                self._cfg.args.input_dim, self._cfg.args.num_classes),
            Est=DensityEstimator(self._cfg.args.style_dim, self._cfg.args.class_dim))

    def _set_criterions(self):
        self._criterions['dec'] = CrossEntropyLoss(lmd=self._cfg.args.lambda_dec)
        self._criterions['rec'] = RecLoss(lmd=self._cfg.args.lambda_rec)
        self._criterions['est'] = EstLoss(radius=self._cfg.args.emb_radius)

    def _set_optimizers(self):
        self._optimizers['main'] = torch.optim.Adam(
            list(self._Enc_style.parameters()) + list(self._Enc_class.parameters()) +
            list(self._Dec.parameters()) + list(self._Rec.parameters()),
            lr=self._cfg.args.learning_rate, betas=(0.5, 0.999))
        self._optimizers['est'] = torch.optim.Adam(
            self._Est.parameters(), lr=self._cfg.args.learning_rate, betas=(0.5, 0.999))

    def _set_logs(self, **kwargs):
        super(DisenIB, self)._set_logs()

        def _get_logger(_name):
            return Logger(
                self._cfg.args.ana_dir, _name,
                formatted_prefix=self._cfg.args.desc, formatted_counters=['epoch', 'step', 'iter'],
                append_mode=False if self._cfg.args.load_from == -1 else True)

        # Evaluation
        self._logs['log_eval_acc'] = _get_logger('eval_accuracy')
        self._logs['log_eval_mi'] = _get_logger('eval_mi')
        self._logs['log_eval_attack'] = _get_logger('eval_attack_robustness')
        self._logs['log_eval_detection'] = _get_logger('eval_out_detection')

    def _set_meters(self, **kwargs):
        super(DisenIB, self)._set_meters()
        self._meters['counter_eval_quant'] = FreqCounter(self._cfg.args.freq_step_eval_quant)
        self._meters['trigger_est'] = TriggerLambda(lambda n: n >= self._cfg.args.est_thr)
        self._meters['trigger_est_style_optimize'] = TriggerPeriod(
            period=self._cfg.args.est_style_optimize + 1, area=self._cfg.args.est_style_optimize)

    def _deploy_batch_data(self, batch_data):
        x, label = map(lambda _x: _x.to(self._cfg.args.device), batch_data)
        return x.size(0), (x, label)

    def _train_step(self, packs):
        _resampling = partial(resampling, n_samples=self._cfg.args.n_samples)
        _repeat = partial(repeat, num=self._cfg.args.n_samples)
        ################################################################################################################
        # Main
        ################################################################################################################
        for _ in range(self._cfg.args.n_times_main):
            x, label = self._fetch_batch_data()
            # Clear grad
            set_requires_grad([self._Enc_style, self._Enc_class, self._Dec, self._Rec], requires_grad=True)
            set_requires_grad([self._Est], requires_grad=False)
            self._optimizers['main'].zero_grad()
            # ----------------------------------------------------------------------------------------------------------
            # Decoding & reconstruction
            # ----------------------------------------------------------------------------------------------------------
            # 1. Decoding
            style_emb, class_emb = self._Enc_style(x), self._Enc_class(x)
            dec_output = self._Dec(_resampling(class_emb, self._cfg.args.class_std))
            loss_dec = self._criterions['dec'](dec_output, _repeat(label))
            # 2. Reconstruction
            rec_output = self._Rec(_resampling(style_emb, self._cfg.args.style_std), _repeat(label))
            loss_rec = self._criterions['rec'](rec_output, _repeat(x))
            # Backward
            summarize_losses_and_backward(loss_dec, loss_rec, retain_graph=True)
            # ----------------------------------------------------------------------------------------------------------
            # Estimator
            # ----------------------------------------------------------------------------------------------------------
            # Calculate output (batch*n_samples, ) & loss (1, )
            est_output = self._Est(
                _resampling(style_emb, self._cfg.args.est_style_std),
                _resampling(class_emb, self._cfg.args.est_class_std), mode='orig')
            crit_est = self._criterions['est'](
                output=est_output, emb=(style_emb, class_emb), mode='main',
                lmd={'loss_est': self._cfg.args.lambda_est, 'loss_wall': self._cfg.args.lambda_wall})
            # Backward
            # 1> Density estimation
            if self._meters['trigger_est'].check(self._meters['i']['step']):
                if self._meters['trigger_est_style_optimize'].check():
                    set_requires_grad(self._Enc_class, requires_grad=False)
                    summarize_losses_and_backward(crit_est['loss_est'], retain_graph=True)
                    set_requires_grad(self._Enc_class, requires_grad=True)
                else:
                    set_requires_grad(self._Enc_style, requires_grad=False)
                    summarize_losses_and_backward(crit_est['loss_est'], retain_graph=True)
                    set_requires_grad(self._Enc_style, requires_grad=True)
            # 2> Embedding wall
            summarize_losses_and_backward(crit_est['loss_wall'], retain_graph=True)
            # ----------------------------------------------------------------------------------------------------------
            # Update
            self._optimizers['main'].step()
            """ Saving """
            packs['log'].update({
                # Decoding & reconstruction
                'loss_dec': loss_dec.item(), 'loss_rec': loss_rec.item(),
                # Estimator
                'loss_est_NO_DISPLAY': crit_est['loss_est'].item(), 'est': crit_est['est'].item()
            })
        ################################################################################################################
        # Density Estimator
        ################################################################################################################
        for _ in range(self._cfg.args.n_times_est):
            with self._meters['timers']('io'):
                x, label = map(lambda _x: _x.to(self._cfg.args.device), next(self._data['train_est']))
            # Clear grad
            set_requires_grad([self._Enc_style, self._Enc_class, self._Dec, self._Rec], requires_grad=False)
            set_requires_grad([self._Est], requires_grad=True)
            self._optimizers['est'].zero_grad()
            # 1. Get embedding
            style_emb, class_emb = self._Enc_style(x).detach(), self._Enc_class(x).detach()
            #   Applying unsupervised clustering to stabilize training
            style_emb_clustered = clustering(style_emb, factor=self._cfg.args.style_std)
            class_emb_clustered = clustering(class_emb, factor=self._cfg.args.class_std)
            # 2. Get output (batch*n_samples, ) & loss (1, ).
            est_output_real = self._Est(
                _resampling(style_emb_clustered, self._cfg.args.est_style_std),
                _resampling(class_emb_clustered, self._cfg.args.est_class_std), mode='perm')
            est_output_fake = self._Est(
                _resampling(style_emb, self._cfg.args.est_style_std),
                _resampling(class_emb, self._cfg.args.est_class_std), mode='orig')
            crit_est = self._criterions['est'](
                output_fake=est_output_fake, output_real=est_output_real, mode='est',
                lmd={'loss_real': 1.0, 'loss_fake': 1.0, 'loss_zc': self._cfg.args.lambda_est_zc})
            # Backward & update
            summarize_losses_and_backward(crit_est['loss_real'], crit_est['loss_fake'], crit_est['loss_zc'])
            self._optimizers['est'].step()
            """ Saving """
            packs['log'].update({
                # Anchor
                'loss_est_real_NO_DISPLAY': crit_est['loss_real'].item(), 'est_real': crit_est['est_real'].item(),
                'loss_est_fake_NO_DISPLAY': crit_est['loss_fake'].item(), 'est_fake': crit_est['est_fake'].item()})

    def _process_after_step(self, packs, **kwargs):
        # 1. Logging
        self._process_log_after_step(packs)
        # 2. Evaluation
        if self._meters['counter_eval_quant'].check(self._meters['i']['step']):
            # (1) Accuracy
            self._api_eval_accuracy(self._data['eval_train'], self._data['eval_test'])
            # (2) Robustness
            self._api_eval_attack_robustness(self._data['eval_train'], self._data['eval_test'])
            # (3) Out detection
            self._api_eval_out_detection(
                self._data['eval_test'], self._cfg.args.eval_odin_out_data, out_name=self._cfg.args.eval_odin_out_data)
            # (4) Mutual information
            self._api_eval_mi(self._data['eval_train'], self._data['eval_test'])
        # 3. Chkpt
        self._process_chkpt_and_lr_after_step()
        # Clear packs
        packs['log'] = ValidContainer()

    def _process_log_after_step(self, packs, **kwargs):

        def _lmd_generate_log():
            r_tfboard = {
                'train/losses': fet_d(packs['log'], prefix='loss_', remove=('loss_', '_NO_DISPLAY')),
                'train/est': fet_d(packs['log'], prefix='est_')
            }
            packs['log'] = packs['log'].dict
            packs['tfboard'] = r_tfboard

        super(DisenIB, self)._process_log_after_step(
            packs, lmd_generate_log=_lmd_generate_log, lmd_process_log=Logger.reform_no_display_items)

    ####################################################################################################################
    # Evaluation
    ####################################################################################################################

    def _api_eval_accuracy(self, train_dataloader, test_dataloader):
        evaluator = AccuracyEvaluator(self._Enc_class, self._Dec, device=self._cfg.args.device)
        # 1. Evaluating on train & test dataloader
        train_acc = evaluator(train_dataloader)
        test_acc = evaluator(test_dataloader)
        # 2. Logging
        self._logs['log_eval_acc'].info_formatted(
            fet_d(self._meters['i'], 'epoch', 'step', 'iter'), items={'train_acc': train_acc, 'test_acc': test_acc})
        tfboard_add_multi_scalars(
            self._logs['tfboard'], multi_scalars={
                'eval/acc': {'train': train_acc, 'test': test_acc}
            }, global_step=self._meters['i']['iter'])
        # Return
        return test_acc

    def _api_eval_mi(self, train_dataloader, test_dataloader):

        def _func_encode_class(_x):
            _class_emb = self._Enc_class(_x)
            _std = self._cfg.args.class_std * torch.ones(size=_class_emb.size(), device=_class_emb.device)
            return {'emb': _class_emb, 'params': (_class_emb, _std)}

        def _func_encode_style(_x):
            _style_emb = self._Enc_style(_x)
            _std = self._cfg.args.style_std * torch.ones(size=_style_emb.size(), device=_style_emb.device)
            return {'emb': _style_emb, 'params': (_style_emb, _std)}

        evaluator_class = MIEvaluator(_func_encode_class, self._Dec, device=self._cfg.args.device)
        evaluator_style = MIEvaluator(_func_encode_style, self._Dec, device=self._cfg.args.device)
        # 1. Evaluating on train & test dataloader
        ret = {}
        for d_name, dataloader in zip(['train', 'test'], [train_dataloader, test_dataloader]):
            ret['%s_mi_x_class' % d_name] = evaluator_class.eval_mi_x_z_monte_carlo(dataloader)
            ret['%s_mi_y_class' % d_name] = evaluator_class.eval_mi_y_z_variational_lb(dataloader)
            ret['%s_mi_x_style' % d_name] = evaluator_style.eval_mi_x_z_monte_carlo(dataloader)
        # 2. Logging
        self._logs['log_eval_mi'].info_formatted(fet_d(self._meters['i'], 'epoch', 'step', 'iter'), items=ret)
        tfboard_add_multi_scalars(
            self._logs['tfboard'], multi_scalars={
                'eval/mi_x_class': {d_name: ret['%s_mi_x_class' % d_name] for d_name in ['train', 'test']},
                'eval/mi_y_class': {d_name: ret['%s_mi_y_class' % d_name] for d_name in ['train', 'test']},
                'eval/mi_x_style': {d_name: ret['%s_mi_x_style' % d_name] for d_name in ['train', 'test']}
            }, global_step=self._meters['i']['iter'])

    def _api_eval_attack_robustness(self, train_dataloader, test_dataloader):
        torch.cuda.empty_cache()
        # 1. Evaluating
        evaluator = AdvAttackEvaluator(
            self._Enc_class, self._Dec, device=self._cfg.args.device, epsilon_list=self._cfg.args.eval_attack_epsilons)
        train_attack_acc = evaluator(train_dataloader)
        test_attack_acc = evaluator(test_dataloader)
        # 2. Logging
        items = {'train_%s' % k: v for k, v in train_attack_acc.items()}
        items.update({'test_%s' % k: v for k, v in test_attack_acc.items()})
        self._logs['log_eval_attack'].info_formatted(fet_d(self._meters['i'], 'epoch', 'step', 'iter'), items=items)
        tfboard_add_multi_scalars(
            self._logs['tfboard'], multi_scalars={
                'eval/attack_robustness': items
            }, global_step=self._meters['i']['iter'])

    def _api_eval_out_detection(self, test_dataloader, out_dataloader, out_name):
        torch.cuda.empty_cache()
        # 1. Evaluating
        evaluator = OutDetectionEvaluator(
            self._Enc_class, self._Dec, device=self._cfg.args.device,
            temper=self._cfg.args.eval_odin_temper, noise_magnitude=self._cfg.args.eval_odin_noise_mag,
            num_delta=self._cfg.args.eval_odin_num_delta)
        scores = evaluator(in_dataloader=test_dataloader, out_dataloader=out_dataloader)
        # 2. Logging
        self._logs['log_eval_detection'].info_formatted(
            fet_d(self._meters['i'], 'epoch', 'step', 'iter'), items={'%s_%s' % (out_name, k): v for k, v in scores.items()})
        tfboard_add_multi_scalars(
            self._logs['tfboard'], multi_scalars={
                'eval/out_detection_%s' % out_name: scores
            }, global_step=self._meters['i']['iter'])

