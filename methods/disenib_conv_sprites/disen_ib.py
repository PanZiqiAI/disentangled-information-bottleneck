
import os
import numpy as np
from shared_libs.modellib.conv import *
from shared_libs.utils.operations import resampling
from shared_libs.utils.evaluations import vis_grid_disentangling
from shared_libs.custom_packages.custom_io.logger import Logger
from shared_libs.dataset_apis.disentangling.datasets import Sprites
from shared_libs.utils.criterions import CrossEntropyLoss, RecLoss, EstLoss
from shared_libs.custom_packages.custom_pytorch.base_models import IterativeBaseModel
from shared_libs.custom_packages.custom_basic.operations import fet_d, ValidContainer
from shared_libs.custom_packages.custom_basic.metrics import FreqCounter, TriggerPeriod, TriggerLambda
from shared_libs.custom_packages.custom_pytorch.operations import summarize_losses_and_backward, set_requires_grad


class DisenIB(IterativeBaseModel):
    """
    Disentangled IB model (conv).
    """
    def _build_architectures(self, **modules):
        super(DisenIB, self)._build_architectures(
            Enc_style=EncoderSprites(self._cfg.args.style_dim), Enc_class=EncoderSprites(self._cfg.args.class_dim),
            Dec=Decoder(self._cfg.args.class_dim, self._cfg.args.num_classes),
            Rec=ReconstructorSprites(self._cfg.args.style_dim, self._cfg.args.class_dim, self._cfg.args.num_classes),
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

    def _set_meters(self, **kwargs):
        super(DisenIB, self)._set_meters()
        self._meters['counter_eval'] = FreqCounter(self._cfg.args.freq_step_eval)
        self._meters['trigger_est'] = TriggerLambda(lambda n: n >= self._cfg.args.est_thr)
        self._meters['trigger_est_style_optimize'] = TriggerPeriod(
            period=self._cfg.args.est_style_optimize + 1, area=self._cfg.args.est_style_optimize)
        # Disentangling debug batch data
        dataset = Sprites()
        self._meters['disen_debug'] = {
            'data': np.concatenate([
                dataset.subset_with_structured_factors({'instance': [0], 'viewpoint': [3], 'frame': [3]})['all'],
                dataset.subset_with_structured_factors({'instance': [1], 'viewpoint': [3], 'frame': [1]})['all'],
                dataset.subset_with_structured_factors({'instance': [2], 'viewpoint': [3], 'frame': [46]})['all'],
                dataset.subset_with_structured_factors({'instance': [3], 'viewpoint': [1], 'frame': [64]})['all'],
                dataset.subset_with_structured_factors({'instance': [4], 'viewpoint': [1], 'frame': [30]})['all'],
                dataset.subset_with_structured_factors({'instance': [5], 'viewpoint': [3], 'frame': [33]})['all'],
                dataset.subset_with_structured_factors({'instance': [6], 'viewpoint': [0], 'frame': [65]})['all'],
                dataset.subset_with_structured_factors({'instance': [7], 'viewpoint': [2], 'frame': [6]})['all']]),
            'label': np.array(list(range(self._cfg.args.num_classes)), dtype=np.int64)}

    # ------------------------------------------------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------------------------------------------------

    def _deploy_batch_data(self, batch_data):
        image, label = map(lambda x: x.to(self._cfg.args.device), batch_data)
        return image.size(0), (image, label)

    def _train_step(self, packs):
        ################################################################################################################
        # Main
        ################################################################################################################
        for _ in range(self._cfg.args.n_times_main):
            images, label = self._fetch_batch_data()
            # Clear grad
            set_requires_grad([self._Enc_style, self._Enc_class, self._Dec, self._Rec], requires_grad=True)
            set_requires_grad([self._Est], requires_grad=False)
            self._optimizers['main'].zero_grad()
            # ----------------------------------------------------------------------------------------------------------
            # Decoding & reconstruction
            # ----------------------------------------------------------------------------------------------------------
            # 1. Decoding
            style_emb, class_emb = self._Enc_style(images), self._Enc_class(images)
            dec_output = self._Dec(resampling(class_emb, self._cfg.args.class_std))
            loss_dec = self._criterions['dec'](dec_output, label)
            # 2. Reconstruction
            rec_output = self._Rec(resampling(style_emb, self._cfg.args.style_std), label)
            loss_rec = self._criterions['rec'](rec_output, images)
            # Backward
            summarize_losses_and_backward(loss_dec, loss_rec, retain_graph=True)
            # ----------------------------------------------------------------------------------------------------------
            # Estimator
            # ----------------------------------------------------------------------------------------------------------
            # Calculate output (batch*n_samples, ) & loss (1, ).
            est_output = self._Est(
                resampling(style_emb, self._cfg.args.est_style_std),
                resampling(class_emb, self._cfg.args.est_class_std), mode='orig')
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
                images, label = map(lambda _x: _x.to(self._cfg.args.device), next(self._data['train_est']))
            # Clear grad
            set_requires_grad([self._Enc_style, self._Enc_class, self._Dec, self._Rec], requires_grad=False)
            set_requires_grad(self._Est, requires_grad=True)
            self._optimizers['est'].zero_grad()
            # 1. Get embedding
            style_emb, class_emb = self._Enc_style(images).detach(), self._Enc_class(images).detach()
            # 2. Get output (batch*n_samples, ) & loss (1, ).
            est_output_real = self._Est(
                resampling(style_emb, self._cfg.args.est_style_std),
                resampling(class_emb, self._cfg.args.est_class_std), mode='perm')
            est_output_fake = self._Est(
                resampling(style_emb, self._cfg.args.est_style_std),
                resampling(class_emb, self._cfg.args.est_class_std), mode='orig')
            crit_est = self._criterions['est'](
                output_fake=est_output_fake, output_real=est_output_real, mode='est',
                lmd={'loss_real': 1.0, 'loss_fake': 1.0, 'loss_zc': self._cfg.args.lambda_est_zc})
            # Backward
            summarize_losses_and_backward(crit_est['loss_real'], crit_est['loss_fake'], crit_est['loss_zc'])
            # Update
            self._optimizers['est'].step()
            """ Saving """
            packs['log'].update({
                # Anchor
                'loss_est_real_NO_DISPLAY': crit_est['loss_real'].item(), 'est_real': crit_est['est_real'].item(),
                'loss_est_fake_NO_DISPLAY': crit_est['loss_fake'].item(), 'est_fake': crit_est['est_fake'].item()})

    def _process_after_step(self, packs, **kwargs):
        # 1. Logging
        self._process_log_after_step(packs)
        # 2. Disentangling
        if self._meters['counter_eval'].check(self._meters['i']['step']):
            # Randomly generating
            vis_grid_disentangling(
                batch_data=map(lambda x: x[:self._cfg.args.eval_dis_n_samples], self._fetch_batch_data(no_record=True)),
                func_style=self._Enc_style, func_rec=self._Rec, gap_size=3,
                save_path=os.path.join(self._cfg.args.eval_dis_dir, 'step[%d].png' % self._meters['i']['step']))
            # Debugging
            vis_grid_disentangling(
                batch_data=(torch.tensor(self._meters['disen_debug']['data'], device=self._cfg.args.device),
                            torch.tensor(self._meters['disen_debug']['label'], device=self._cfg.args.device)),
                func_style=self._Enc_style, func_rec=self._Rec, gap_size=3,
                save_path=os.path.join(self._cfg.args.eval_dis_dir, 'debug[%d].png' % self._meters['i']['step']))
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
