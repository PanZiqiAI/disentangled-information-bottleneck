
import math
import copy
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
from sklearn.metrics import accuracy_score
from shared_libs.utils.operations import gaussian_log_density_marginal
from shared_libs.custom_packages.custom_io.logger import show_progress


# ----------------------------------------------------------------------------------------------------------------------
# Generalization
# ----------------------------------------------------------------------------------------------------------------------

class AccuracyEvaluator(object):
    """
    Evaluating generalization.
    """
    def __init__(self, encoder, decoder, device):
        # Modules
        self._Enc, self._Dec = encoder, decoder
        # Config
        self._device = device

    @torch.no_grad()
    def __call__(self, dataloader):
        torch.cuda.empty_cache()
        # 1. Calculate prediction
        pred, gt = [], []
        for batch_index, (images, label) in enumerate(dataloader):
            show_progress("Evaluating accuracy", index=batch_index, maximum=len(dataloader))
            # 1. Decoding
            dec_output = self._Dec(self._Enc(images.to(self._device)))
            # 2. Get predicted
            cur_pred = torch.argmax(dec_output, dim=1)
            # Save
            pred.append(cur_pred.detach().cpu().numpy())
            gt.append(label.detach().cpu().numpy())
        pred, gt = np.concatenate(pred), np.concatenate(gt)
        # 2. Calculate accuracy
        acc = accuracy_score(gt, pred)
        # Return
        return acc


# ----------------------------------------------------------------------------------------------------------------------
# Visualizing disentangling
# ----------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def vis_grid_disentangling(batch_data, func_style, func_rec, gap_size, save_path):
    """
    Visualizing disentangling in grid.
    """
    images, class_label = batch_data
    # 1. Calculate reconstructions
    # (1) Encoded & get shape. (batch, style_dim)
    style_mu = func_style(images)
    batch, style_dim = style_mu.size()
    # (2) Mesh grid. (batch*batch, style_dim) & (batch*batch, class_dim)
    style_mu = style_mu.unsqueeze(1).expand(batch, batch, style_dim).reshape(-1, style_dim)
    class_label = class_label.unsqueeze(0).expand(batch, batch).reshape(-1, )
    # (3) Decode. (batch*batch, ...)
    recon = func_rec(style_mu, class_label)
    # 2. Get result
    recon = torch.reshape(recon, shape=(batch, batch, *recon.size()[1:]))
    recon = torch.cat([_.squeeze(1) for _ in torch.split(recon, split_size_or_sections=1, dim=1)], dim=3)
    recon = torch.cat([_.squeeze(0) for _ in torch.split(recon, split_size_or_sections=1, dim=0)], dim=1)
    # 1> Right
    hor_images = torch.cat([_.squeeze(0) for _ in torch.split(images, split_size_or_sections=1, dim=0)], dim=2)
    hor_gap = torch.ones(size=(hor_images.size(0), gap_size, hor_images.size(2)), device=hor_images.device)
    ret = torch.cat([hor_images, hor_gap, recon], dim=1)
    # 2> Left
    ver_images = torch.cat([_.squeeze(0) for _ in torch.split(images, split_size_or_sections=1, dim=0)], dim=1)
    ver_images = torch.cat([
        torch.zeros(size=(ver_images.size(0), images.size(2), images.size(3)), device=ver_images.device),
        torch.ones(size=(ver_images.size(0), gap_size, images.size(3)), device=ver_images.device),
        ver_images], dim=1)
    ver_gap = torch.ones(size=(ver_images.size(0), ver_images.size(1), gap_size), device=ver_images.device)
    ver_images = torch.cat([ver_images, ver_gap], dim=2)
    # Result
    ret = torch.cat([ver_images, ret], dim=2)
    # 3. Save
    save_image(ret.unsqueeze(0), save_path)


# ----------------------------------------------------------------------------------------------------------------------
# Estimating Mutual Information
# ----------------------------------------------------------------------------------------------------------------------

class MIEvaluator(object):
    """
    Mutual Information estimator.
    """
    def __init__(self, encoder, decoder, device):
        """
        - func_encode:
            kwargs: x (batch, ...)
            output: {'emb': (batch, nz), 'param': (mu: (batch, nz), std: (batch, nz))}
        - func_decode:
            kwargs: z (batch, nz)
            output: (batch, num_classes)
        """
        # Modules
        self._Enc, self._Dec = encoder, decoder
        # Config
        self._device = device

    @torch.no_grad()
    def eval_mi_x_z_monte_carlo(self, dataloader):
        torch.cuda.empty_cache()
        # Get dataloaders
        x_dataloader = dataloader
        z_dataloader = copy.deepcopy(dataloader)
        ################################################################################################################
        # Eval H(X|Z)
        ################################################################################################################
        ent_x_z = []
        for batch_z_index, batch_z_data in enumerate(z_dataloader):
            # Get z. (batch, nz)
            batch_z = self._Enc(batch_z_data[0].to(self._device))['emb']
            ############################################################################################################
            # Calculate H(X|batch_z)
            ############################################################################################################
            # 1. Get log p(batch_z|x). (batch, total_num_x)
            log_p_batch_z_x = []
            for batch_x_index, batch_x_data in enumerate(x_dataloader):
                show_progress(
                    "Estimating I(X;Z)", index=batch_z_index * len(x_dataloader) + batch_x_index,
                    maximum=len(x_dataloader) * len(x_dataloader))
                # (1) Get params (mu, std). (batch, nz)
                batch_x = self._Enc(batch_x_data[0].to(self._device))['params']
                # (2) Get log p(batch_z|batch_x). (batch, batch)
                log_p_batch_z_batch_x = gaussian_log_density_marginal(batch_z, batch_x, mesh=True).sum(dim=2)
                # Accumulate
                log_p_batch_z_x.append(log_p_batch_z_batch_x)
            log_p_batch_z_x = torch.cat(log_p_batch_z_x, dim=1)
            # 2. Normalize to get log p(x|batch_z). (batch, total_num_x)
            log_p_x_batch_z = log_p_batch_z_x - torch.logsumexp(log_p_batch_z_x, dim=1, keepdim=True)
            # 3. Get H(X|batch_z). (batch, )
            ent_x_batch_z = (-torch.exp(log_p_x_batch_z) * log_p_x_batch_z).sum(dim=1)
            # Accumulate
            ent_x_z.append(ent_x_batch_z)
        ent_x_z = torch.cat(ent_x_z, dim=0).mean()
        ################################################################################################################
        # Eval H(X)
        ################################################################################################################
        ent_x = math.log(len(x_dataloader.dataset))
        ################################################################################################################
        # Eval I(X;Z) = H(X) - H(X|Z)
        ################################################################################################################
        ret = ent_x - ent_x_z
        # Return
        return ret

    @torch.no_grad()
    def eval_mi_y_z_variational_lb(self, dataloader):
        torch.cuda.empty_cache()
        ####################################################################################################################
        # Eval H(Y|Z) upper bound.
        ####################################################################################################################
        ent_y_z = []
        for batch_index, batch_data in enumerate(dataloader):
            show_progress("Estimating I(Z;Y)", batch_index, len(dataloader))
            # (1) Get image & label, embedding (batch, nz)
            batch_x, label = map(lambda _x: _x.to(self._device), batch_data)
            batch_z = self._Enc(batch_x)['emb']
            # (2) Get H(Y|batch_z). (batch, )
            prob = torch.softmax(self._Dec(batch_z), dim=1)
            ent_y_batch_z = (-prob * torch.log(prob + 1e-10)).sum(dim=1)
            # Accumulate to result
            ent_y_z.append(ent_y_batch_z)
        ent_y_z = torch.cat(ent_y_z, dim=0).mean()
        ####################################################################################################################
        # Get H(Y)
        ####################################################################################################################
        class_counter = np.array(dataloader.dataset.class_counter)
        class_prob = class_counter / class_counter.sum()
        ent_y = (-class_prob * np.log(class_prob)).sum()
        ####################################################################################################################
        # Eval I(Y;Z) = H(Y) - H(Y|Z)
        ####################################################################################################################
        ret = ent_y - ent_y_z
        # Return
        return ret


# ----------------------------------------------------------------------------------------------------------------------
# Robustness to adversary attack
# ----------------------------------------------------------------------------------------------------------------------

class AdvAttackEvaluator(object):
    """
    Evaluating robustness to adversary attack using FGSM.
    """
    def __init__(self, encoder, decoder, device, epsilon_list):
        # Modules
        self._Enc, self._Dec = encoder, decoder
        # Configs
        self._device = device
        self._epsilon_list = epsilon_list

    def _get_output(self, x, mode, **kwargs):
        assert mode in ['output', 'pred', 'acc']
        output = self._Dec(self._Enc(x))
        if mode == 'output': return output
        pred = output.max(dim=1, keepdim=True)[1].squeeze()
        if mode == 'pred': return pred
        acc = pred == kwargs['label']
        return acc

    def _perturb_image_fgsm(self, image, label):
        image.requires_grad_(True)
        # 1. Calculate output & value
        output = self._get_output(image, mode='output')
        loss_value = F.nll_loss(output, label)
        # 2. Calculate gradient
        self._Enc.zero_grad()
        self._Dec.zero_grad()
        loss_value.backward()
        # 3. Take gradient sign
        image_grad_sign = image.grad.data.sign()
        perturbed_image_list = [torch.clamp(image + eps * image_grad_sign, min=-1, max=1) for eps in self._epsilon_list]
        # Return
        return perturbed_image_list

    def __call__(self, dataloader):
        attack_acc_list = [[] for _ in self._epsilon_list]
        for batch_index, batch_data in enumerate(dataloader):
            show_progress("Evaluating robustness to adversary attack", batch_index, len(dataloader))
            image, label = map(lambda _x: _x.to(self._device), batch_data)
            # 1. Get correct
            batch_acc_init = self._get_output(image, mode='acc', label=label)
            # (1) Get indices
            batch_indices = np.argwhere(batch_acc_init.cpu().numpy())[:, 0]
            batch_indices = None if len(batch_indices) == 0 else torch.LongTensor(batch_indices).to(self._device)
            if batch_indices is None: continue
            # (2) Get samples & labels
            correct_image = torch.index_select(image, dim=0, index=batch_indices)
            correct_label = torch.index_select(label, dim=0, index=batch_indices)
            # 2. Get perturbed images
            perturbed_image_list = self._perturb_image_fgsm(correct_image, correct_label)
            # 3. Re-classify perturbed image
            batch_perturbed_acc_list = [self._get_output(perturbed_image, label=correct_label, mode='acc')
                                        for perturbed_image in perturbed_image_list]
            # Save
            for eps_index, batch_perturbed_acc in enumerate(batch_perturbed_acc_list):
                attack_acc_list[eps_index].append(batch_perturbed_acc)
        attack_acc_list = [torch.cat(aa, dim=0).float().mean() for aa in attack_acc_list]
        # Return
        return {'eps_%.1f' % eps: aa for eps, aa in zip(self._epsilon_list, attack_acc_list)}


# ----------------------------------------------------------------------------------------------------------------------
# Out-of-distribution detection
# ----------------------------------------------------------------------------------------------------------------------

def tpr95(in_softmax_scores, out_softmax_scores):
    """
    False Positive Rate (FPR) when TPR == 95%.
    :return:
    """
    # 1. Init result & counter
    result, counter = 0.0, 0
    # 2. Traverse delta
    # (1) Get delta_list
    reversed_in_softmax_scores = np.sort(in_softmax_scores)[::-1]
    upper_num = int(0.9505 * len(reversed_in_softmax_scores))
    lower_num = int(0.9495 * len(reversed_in_softmax_scores))
    if upper_num == lower_num:
        delta_list = [reversed_in_softmax_scores[upper_num]]
    else:
        delta_list = reversed_in_softmax_scores[lower_num:upper_num]
    # (2) Traversing
    for delta in delta_list:
        fpr = np.sum(out_softmax_scores >= delta) * 1.0 / len(out_softmax_scores)
        result += fpr
        counter += 1
    # 3. Get result
    result = result / counter
    # Return
    return result


def auroc(in_softmax_scores, out_softmax_scores, num_delta):
    """
    Area Under the Receiver Operating Characteristic Curve (AUROC). The ROC curve (FPR, TPR)
    :return:
    """
    # 1. Init
    result = 0.0
    # 2. Approximating Calculus
    # (1) Init last_fpr
    last_fpr = 1.0
    # (2) Traversing delta (different points on ROC curve)
    # Get delta_start & delta_end
    delta_start = np.minimum(np.min(in_softmax_scores), np.min(out_softmax_scores))
    delta_end = np.maximum(np.max(in_softmax_scores), np.max(out_softmax_scores))
    delta_gap = (delta_end - delta_start) / num_delta
    # Traversing
    for delta in np.arange(delta_start, delta_end, delta_gap):
        tpr = np.sum(in_softmax_scores >= delta) / len(in_softmax_scores)
        fpr = np.sum(out_softmax_scores >= delta) / len(out_softmax_scores)
        result += (last_fpr - fpr) * tpr
        last_fpr = fpr
    # Return
    return result


def aupr(in_softmax_scores, out_softmax_scores, mode, num_delta):
    """
    Area Under Precision Recall curve.
    :return:
    """
    assert mode in ['in', 'out']
    # 1. Init result
    result = 0.0
    # 2. Approximating calculus
    # (1) Init last_recall
    last_recall = 1.0
    # (2) Traversing delta
    # Get delta_start & delta_end
    delta_start = np.minimum(np.min(in_softmax_scores), np.min(out_softmax_scores))
    delta_end = np.maximum(np.max(in_softmax_scores), np.max(out_softmax_scores))
    delta_gap = (delta_end - delta_start) / num_delta
    # Traversing
    for delta in np.arange(delta_start, delta_end, delta_gap) if mode == 'in' else \
            np.arange(delta_end, delta_start, -delta_gap):
        # 1. Precision & recall (tp)
        if mode == 'in':
            tp = np.sum(in_softmax_scores >= delta) / len(in_softmax_scores)
            fp = np.sum(out_softmax_scores >= delta) / len(out_softmax_scores)
        else:
            fp = np.sum(in_softmax_scores < delta) / len(in_softmax_scores)
            tp = np.sum(out_softmax_scores < delta) / len(out_softmax_scores)
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        # 2. Save to result
        result += (last_recall - tp) * precision
        # 3. Update last_recall
        last_recall = tp
    # Return
    return result


def detection_error(in_softmax_scores, out_softmax_scores, num_delta):
    """
    0.5 * (1 - TPR) + 0.5 * FPR
    :return:
    """
    # 1. Init result
    result = 1.0
    # 2. Traversing delta
    # (1) Get delta_start & delta_end
    delta_start = np.minimum(np.min(in_softmax_scores), np.min(out_softmax_scores))
    delta_end = np.maximum(np.max(in_softmax_scores), np.max(out_softmax_scores))
    delta_gap = (delta_end - delta_start) / num_delta
    # (2) Traversing
    for delta in np.arange(delta_start, delta_end, delta_gap):
        tpr = np.sum(in_softmax_scores >= delta) / len(in_softmax_scores)
        fpr = np.sum(out_softmax_scores >= delta) / len(out_softmax_scores)
        result = np.minimum(result, (1.0 - tpr + fpr) / 2.0)
    # Return
    return result


class OutDetectionEvaluator(object):
    """
    Out-of-distribution detection using ODIN.
    """
    def __init__(self, encoder, decoder, device, temper, noise_magnitude, num_delta):
        # Modules
        self._Enc, self._Dec = encoder, decoder
        # Configs
        self._device = device
        self._temper = temper
        self._noise_magnitude = noise_magnitude
        self._num_delta = num_delta

    def _get_output(self, image):
        output = self._Dec(self._Enc(image))
        # Return
        return output / self._temper

    def _out_detection_odin_procedure(self, image):
        # Set gradient
        image.requires_grad_(True)
        # 1. Get initial output using temperature scaling.
        output = self._get_output(image)
        pred = F.softmax(output, dim=1).max(dim=1)[1]
        # 2. Get perturbed input
        # (1) Calculate loss & backward
        loss = nn.CrossEntropyLoss()(output, pred)
        loss.backward()
        # (2) Normalize gradient to {-1, 1} & apply to original image tensor
        gradient = (torch.ge(image.grad.data, 0).float() - 0.5) * 2
        perturbed_image_tensor = (image - self._noise_magnitude * gradient).clamp(-1, 1)
        # (3) Get perturbed output using temperature scaling.
        output = self._get_output(perturbed_image_tensor)
        softmax_score = F.softmax(output, dim=1).max(dim=1)[0]
        # Return
        return softmax_score

    def __call__(self, in_dataloader, out_dataloader):
        softmax_scores = {'in': [], 'out': []}
        # --------------------------------------------------------------------------------------------------------------
        # Using Gaussian noise as out-distribution data
        # --------------------------------------------------------------------------------------------------------------
        if out_dataloader == 'gauss':
            for batch_index, batch_data in enumerate(in_dataloader):
                show_progress("Detecting in & out-distribution", index=batch_index, maximum=len(in_dataloader))
                # Deploy data
                image = batch_data[0].to(self._device)
                # 1. IN
                softmax_scores['in'].append(self._out_detection_odin_procedure(image).detach().cpu().numpy())
                # 2. OUT
                noise = torch.randn(size=image.size(), device=self._device).clamp(-1, 1)
                softmax_scores['out'].append(self._out_detection_odin_procedure(noise).detach().cpu().numpy())
        # --------------------------------------------------------------------------------------------------------------
        # Using other dataset as out-distribution data
        # --------------------------------------------------------------------------------------------------------------
        else:
            for name, dataloader in zip(['in', 'out'], [in_dataloader, out_dataloader]):
                for batch_index, batch_data in enumerate(dataloader):
                    show_progress("Detecting %s-distribution" % name, index=batch_index, maximum=len(dataloader))
                    # Deploy data
                    image = batch_data[0].to(self._device)
                    # Calculate
                    softmax_scores[name].append(self._out_detection_odin_procedure(image).detach().cpu().numpy())
        # --------------------------------------------------------------------------------------------------------------
        # Concat results
        softmax_scores['in'] = np.concatenate(softmax_scores['in'], axis=0)
        softmax_scores['out'] = np.concatenate(softmax_scores['out'], axis=0)
        # Get metric scores
        return {
            'tpr95': tpr95(softmax_scores['in'], softmax_scores['out']),
            'auroc': auroc(softmax_scores['in'], softmax_scores['out'], self._num_delta),
            'aupr_in': aupr(softmax_scores['in'], softmax_scores['out'], 'in', self._num_delta),
            'aupr_out': aupr(softmax_scores['in'], softmax_scores['out'], 'out', self._num_delta),
            'detect_err': detection_error(softmax_scores['in'], softmax_scores['out'], self._num_delta)
        }
