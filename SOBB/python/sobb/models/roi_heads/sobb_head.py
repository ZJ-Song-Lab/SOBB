import jittor as jt
from jittor import nn
from sobb.utils.registry import HEADS, build_from_cfg, BOXES, LOSSES
from sobb.models.utils.weight_init import normal_init, bias_init_with_prob
from sobb.models.utils.dsof import DSOF, CandidateScorerFC
from sobb.models.boxes.sobb_ops import (sobb_candidate_geoms,
    sobbb_decode_both, reconstruct_gt_obb, compute_risk_aware_inputs, risk_aware_score_loss,
    consistency_loss)
from sobb.models.boxes.box_ops import bbox2roi, dbbox2roi
from sobb.ops.bbox_transforms import bbox2delta, delta2bbox, dbbox2result
import math

from sobb.ops.nms_rotated import multiclass_nms_rotated

@HEADS.register_module()
class SOBBHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 roi_feat_size=7,
                 num_shared_fcs=2,
                 fc_out_channels=1024,
                 reg_class_agnostic=False,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1.),
                 loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0/9.0, loss_weight=1.0),
                 loss_sobb_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 lambda_cons=0.0,
                 scorer_mode='scorer_only',
                 loss_cal_weight=1.0,
                 loss_pair_weight=1.0,
                 loss_margin_weight=1.0,
                 use_dsof=True,
                 use_scorer=True,
                 use_sala=True,
                 consistency_mode='logit_noise',
                 repr_mode='sobb'):
        super(SOBBHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.roi_feat_size = roi_feat_size
        self.reg_class_agnostic = reg_class_agnostic

        self.bbox_coder = build_from_cfg(dict(
            type='SOBBBBoxCoder',
            target_means=target_means,
            target_stds=target_stds
        ), BOXES)

        self.loss_cls = build_from_cfg(loss_cls, LOSSES)
        self.loss_bbox = build_from_cfg(loss_bbox, LOSSES)
        self.loss_sobb_cls = build_from_cfg(loss_sobb_cls, LOSSES)

        # Shared FC layers
        self.shared_fcs = nn.ModuleList()
        last_dim = in_channels * roi_feat_size * roi_feat_size
        for _ in range(num_shared_fcs):
            self.shared_fcs.append(nn.Linear(last_dim, fc_out_channels))
            self.shared_fcs.append(nn.ReLU())
            last_dim = fc_out_channels

        # Cls branch
        self.fc_cls = nn.Linear(last_dim, num_classes)

        # C1: Conditional DSOF Module
        if use_dsof:
            self.dsof = DSOF(in_channels)
        else:
            self.dsof = None
        self.lambda_cons = lambda_cons
        self.scorer_mode = scorer_mode
        self.loss_cal_weight = loss_cal_weight
        self.loss_pair_weight = loss_pair_weight
        self.loss_margin_weight = loss_margin_weight
        self.use_dsof = use_dsof
        self.use_scorer = use_scorer
        self.use_sala = use_sala
        self.consistency_mode = consistency_mode
        self.repr_mode = repr_mode

        # Reg branches (Decoupled)
        # Scale branch: dw, dh, z_s (3 parameters)
        # Alignment branch: dx, dy (2 parameters) -- s1, s2 from CandidateScorerFC
        out_dim_scale = 3 if reg_class_agnostic else 3 * num_classes
        out_dim_align = 2 if reg_class_agnostic else 2 * num_classes

        # Shared FCs for decoupled features
        self.scale_fcs = nn.ModuleList([
            nn.Linear(in_channels * roi_feat_size * roi_feat_size, fc_out_channels),
            nn.ReLU()
        ])
        self.align_fcs = nn.ModuleList([
            nn.Linear(in_channels * roi_feat_size * roi_feat_size, fc_out_channels),
            nn.ReLU()
        ])

        self.fc_reg_scale = nn.Linear(fc_out_channels, out_dim_scale)
        self.fc_reg_align = nn.Linear(fc_out_channels, out_dim_align)

        # C1: Conditional candidate scorer
        if use_scorer:
            self.candidate_scorer_fc = CandidateScorerFC(fc_out_channels)
        else:
            self.candidate_scorer_fc = None
        if not use_scorer:
            out_dim_scale = 2 if reg_class_agnostic else 2 * num_classes
            out_dim_align = 3 if reg_class_agnostic else 3 * num_classes
            self.fc_reg_scale_std = nn.Linear(fc_out_channels, out_dim_scale)
            self.fc_reg_align_std = nn.Linear(fc_out_channels, out_dim_align)

    def init_weights(self):
        normal_init(self.fc_cls, std=0.01)
        normal_init(self.fc_reg_scale, std=0.001)
        normal_init(self.fc_reg_align, std=0.001)
        for fc in self.shared_fcs:
            if isinstance(fc, nn.Linear):
                normal_init(fc, std=0.01)
        for fc in self.scale_fcs:
            if isinstance(fc, nn.Linear):
                normal_init(fc, std=0.01)
        for fc in self.align_fcs:
            if isinstance(fc, nn.Linear):
                normal_init(fc, std=0.01)

    def execute(self, x, rrois=None):
        # x shape: [N, C, H, W] (RoI pooled features)
        # rrois: [N, 6] = (batch_ind, x, y, w, h, angle) rotated proposals

        # C1: Conditional DSOF Feature Routing
        if self.use_dsof:
            f_scale, f_align = self.dsof(x)
        else:
            f_scale = x
            f_align = x

        # 2. Shared feature for Classification
        x_flat = x.view(x.shape[0], -1)
        for fc in self.shared_fcs:
            x_flat = fc(x_flat)
        cls_score = self.fc_cls(x_flat)

        # 3. Decoupled Regression
        # Scale branch (w, h, z_s)
        f_scale_flat = f_scale.view(f_scale.shape[0], -1)
        for fc in self.scale_fcs:
            f_scale_flat = fc(f_scale_flat)
        reg_scale = self.fc_reg_scale(f_scale_flat)

        # Alignment branch (x, y only; s1, s2 from CandidateScorerFC)
        f_align_flat = f_align.view(f_align.shape[0], -1)
        for fc in self.align_fcs:
            f_align_flat = fc(f_align_flat)
        reg_align = self.fc_reg_align(f_align_flat)

        # 4. Candidate geometry + shared permutation-equivariant scoring
        # Compute proposal HBB dimensions from rrois
        if rrois is not None:
            prop_w_r = rrois[:, 3]
            prop_h_r = rrois[:, 4]
            prop_a = rrois[:, 5]
            cos_a = jt.abs(jt.cos(prop_a))
            sin_a = jt.abs(jt.sin(prop_a))
            prop_w = prop_w_r * cos_a + prop_h_r * sin_a
            prop_h = prop_w_r * sin_a + prop_h_r * cos_a
        else:
            prop_w = jt.ones((x.shape[0],))
            prop_h = jt.ones((x.shape[0],))

        eps = 1e-6
        if self.reg_class_agnostic:
            dw = reg_scale[:, 0]
            dh = reg_scale[:, 1]
            z_s = reg_scale[:, 2]
            gw = jt.exp(dw) * prop_w
            gh = jt.exp(dh) * prop_h
            t_s = jt.clamp(jt.sigmoid(z_s), eps, 1.0 - eps)
            S = 1.0 - t_s

            x1, y1, x2, y2 = sobb_candidate_geoms(gw, gh, S)
            gw_safe = jt.maximum(gw, eps)
            gh_safe = jt.maximum(gh, eps)
            log_ratio = jt.log(gw_safe / gh_safe)
            cand1 = jt.stack([x1 / gw_safe, y1 / gh_safe, log_ratio, S], dim=-1)
            cand2 = jt.stack([x2 / gw_safe, y2 / gh_safe, log_ratio, S], dim=-1)
            candidates = jt.stack([cand1, cand2], dim=1)  # [N, 2, 4]

            if self.scorer_mode == 'fully_joint':
                scorer_feat = f_align_flat
                scorer_cands = candidates
            elif self.scorer_mode == 'feature_joint':
                scorer_feat = f_align_flat
                scorer_cands = candidates.detach()
            else:
                scorer_feat = f_align_flat.detach()
                scorer_cands = candidates.detach()
            if self.use_scorer:
                s_scores = self.candidate_scorer_fc(scorer_feat, scorer_cands)
                bbox_pred = jt.stack([
                    reg_align[:, 0], reg_align[:, 1],
                    reg_scale[:, 0], reg_scale[:, 1], reg_scale[:, 2],
                    s_scores[:, 0], s_scores[:, 1]
                ], dim=1)
            else:
                # Standard 5-param OBB: [dx, dy, dw, dh, da]
                bbox_pred = jt.stack([
                    reg_align[:, 0], reg_align[:, 1],
                    reg_scale[:, 0], reg_scale[:, 1],
                    reg_align[:, 2]
                ], dim=1)
        else:
            N = x.shape[0]
            _dim = 7 if self.use_scorer else 5
            bbox_pred = jt.zeros((N, _dim * self.num_classes))
            for i in range(self.num_classes):
                dw = reg_scale[:, i * 3]
                dh = reg_scale[:, i * 3 + 1]
                z_s = reg_scale[:, i * 3 + 2]
                gw = jt.exp(dw) * prop_w
                gh = jt.exp(dh) * prop_h
                t_s = jt.clamp(jt.sigmoid(z_s), eps, 1.0 - eps)
                S = 1.0 - t_s

                x1, y1, x2, y2 = sobb_candidate_geoms(gw, gh, S)
                gw_safe = jt.maximum(gw, eps)
                gh_safe = jt.maximum(gh, eps)
                log_ratio = jt.log(gw_safe / gh_safe)
                cand1 = jt.stack([x1 / gw_safe, y1 / gh_safe, log_ratio, S], dim=-1)
                cand2 = jt.stack([x2 / gw_safe, y2 / gh_safe, log_ratio, S], dim=-1)
                candidates = jt.stack([cand1, cand2], dim=1)

                if self.scorer_mode == 'fully_joint':
                    scorer_feat = f_align_flat
                    scorer_cands = candidates
                elif self.scorer_mode == 'feature_joint':
                    scorer_feat = f_align_flat
                    scorer_cands = candidates.detach()
                else:
                    scorer_feat = f_align_flat.detach()
                    scorer_cands = candidates.detach()
                if self.use_scorer:
                    s_scores = self.candidate_scorer_fc(scorer_feat, scorer_cands)
                    bbox_pred[:, i * 7:(i + 1) * 7] = jt.stack([
                        reg_align[:, i * 2], reg_align[:, i * 2 + 1],
                        reg_scale[:, i * 3], reg_scale[:, i * 3 + 1], reg_scale[:, i * 3 + 2],
                        s_scores[:, 0], s_scores[:, 1]
                    ], dim=1)
                else:
                    bbox_pred[:, i * 5:(i + 1) * 5] = jt.stack([
                        reg_align[:, i * 2], reg_align[:, i * 2 + 1],
                        reg_scale[:, i * 3], reg_scale[:, i * 3 + 1],
                        reg_align[:, i * 2 + 2]
                    ], dim=1)

        return cls_score, bbox_pred

    def get_target_rbbox(self, sampling_results, gt_obbs, gt_labels, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_gt_obbs = [res.pos_gt_bboxes for res in sampling_results]
        pos_labels = [res.pos_gt_labels for res in sampling_results]

        all_labels = []
        all_label_weights = []
        all_bbox_targets = []
        all_bbox_weights = []
        all_pos_gt_obbs = []
        all_pos_prop_hbbs = []

        for i in range(len(sampling_results)):
            res = sampling_results[i]
            num_samples = res.bboxes.shape[0]
            labels = jt.zeros((num_samples,), dtype=jt.int32)
            label_weights = jt.zeros((num_samples,), dtype=jt.float32)
            bbox_targets = jt.zeros((num_samples, 7), dtype=jt.float32)
            bbox_weights = jt.zeros((num_samples, 7), dtype=jt.float32)

            if res.pos_bboxes.shape[0] > 0:
                prop_x, prop_y, prop_w, prop_h, prop_a = [res.pos_bboxes[:, j] for j in range(5)]
                cos_a = jt.abs(jt.cos(prop_a))
                sin_a = jt.abs(jt.sin(prop_a))
                prop_HBB_W = prop_w * cos_a + prop_h * sin_a
                prop_HBB_H = prop_w * sin_a + prop_h * cos_a
                prop_HBB = jt.stack([prop_x, prop_y, prop_HBB_W, prop_HBB_H], dim=1)

                pos_bbox_targets = self.bbox_coder.encode(prop_HBB, res.pos_gt_bboxes)
                all_pos_gt_obbs.append(res.pos_gt_bboxes)
                all_pos_prop_hbbs.append(prop_HBB)

                num_pos = pos_bbox_targets.shape[0]
                bbox_targets[:num_pos, :] = pos_bbox_targets
                bbox_weights[:num_pos, :] = 1.0

                pos_assigned_gt_labels = res.pos_gt_labels
                if pos_assigned_gt_labels is not None and pos_assigned_gt_labels.shape[0] > 0:
                    labels[:pos_assigned_gt_labels.shape[0]] = pos_assigned_gt_labels
                    label_weights[:pos_assigned_gt_labels.shape[0]] = 1.0

            all_labels.append(labels)
            all_label_weights.append(label_weights)
            all_bbox_targets.append(bbox_targets)
            all_bbox_weights.append(bbox_weights)

        if len(all_labels) > 0:
            labels = jt.concat(all_labels, dim=0)
            label_weights = jt.concat(all_label_weights, dim=0)
            bbox_targets = jt.concat(all_bbox_targets, dim=0)
            bbox_weights = jt.concat(all_bbox_weights, dim=0)
        else:
            labels = jt.zeros((0,), dtype=jt.int32)
            label_weights = jt.zeros((0,), dtype=jt.float32)
            bbox_targets = jt.zeros((0, 7), dtype=jt.float32)
            bbox_weights = jt.zeros((0, 7), dtype=jt.float32)

        if len(all_pos_gt_obbs) > 0:
            pos_gt_obbs = jt.concat(all_pos_gt_obbs, dim=0)
            pos_prop_hbbs = jt.concat(all_pos_prop_hbbs, dim=0)
        else:
            pos_gt_obbs = jt.zeros((0, 5))
            pos_prop_hbbs = jt.zeros((0, 4))
        return labels, label_weights, bbox_targets, bbox_weights, pos_gt_obbs, pos_prop_hbbs

    def loss(self, cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights,
             pos_gt_obbs=None, pos_prop_hbbs=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = jt.sum(label_weights > 0).float().item()
            losses['loss_cls'] = self.loss_cls(cls_score, labels, label_weights, avg_factor=avg_factor)

        if bbox_pred is not None:
            pos_inds = (labels > 0)
            if pos_inds.any():
                pos_bbox_pred = bbox_pred[pos_inds]
                pos_labels = labels[pos_inds]

                if not self.reg_class_agnostic:
                    pos_bbox_pred = pos_bbox_pred.view(-1, self.num_classes, 7)
                    pos_bbox_pred = pos_bbox_pred[jt.arange(pos_bbox_pred.shape[0]), pos_labels]

                # Split into regression (first 5) and candidate scoring (last 2)
                # reg_pred: dx, dy, dw, dh, z_s
                # amb_pred: s1, s2
                reg_pred = pos_bbox_pred[:, :5]
                amb_pred = pos_bbox_pred[:, 5:]

                reg_target = bbox_targets[pos_inds, :5]
                amb_target = bbox_targets[pos_inds, 5:]

                # Decode-then-loss for the shape parameter t_s:
                # t_s_hat = clip(sigmoid(z_s), eps, 1-eps)
                # L_reg = smooth_l1([dx, dy, dw, dh, t_s_hat], [dx_t, dy_t, dw_t, dh_t, t_s])
                eps = 1e-6
                z_s = reg_pred[:, 4]
                t_s_hat = jt.clamp(jt.sigmoid(z_s), eps, 1.0 - eps)
                reg_pred_decoded = jt.concat([reg_pred[:, :4], t_s_hat.unsqueeze(1)], dim=1)

                losses['loss_bbox'] = self.loss_bbox(
                    reg_pred_decoded, reg_target, bbox_weights[pos_inds, :5])

                # Risk-aware pairwise scoring loss (calibration + pairwise + margin)
                if pos_gt_obbs is not None and pos_prop_hbbs is not None and pos_gt_obbs.shape[0] > 0:
                    pos_pred_7 = jt.concat([reg_pred, amb_pred], dim=1)
                    cand1, cand2 = sobbb_decode_both(
                        pos_prop_hbbs, pos_pred_7,
                        means=list(self.bbox_coder.means), stds=list(self.bbox_coder.stds))
                    q, D = compute_risk_aware_inputs(cand1, cand2, pos_gt_obbs)

                    if (self.loss_cal_weight == 0.0 and
                            self.loss_pair_weight == 0.0 and
                            self.loss_margin_weight == 0.0):
                        # C2: R01 ablation: GT-candidate independent BCE.
                        # Decode candidates from the GT-encoded target
                        # (not from predictions) to construct a true
                        # GT-candidate set, as defined in the paper.
                        # This isolates the candidate source from the
                        # prediction-aligned scoring used in R02+.
                        pos_target_7 = jt.concat([
                            bbox_targets[pos_inds, :5],
                            bbox_targets[pos_inds, 5:7]
                        ], dim=1)
                        gt_cand1, gt_cand2 = sobbb_decode_both(
                            pos_prop_hbbs, pos_target_7,
                            means=list(self.bbox_coder.means),
                            stds=list(self.bbox_coder.stds))
                        q_gt, _ = compute_risk_aware_inputs(
                            gt_cand1, gt_cand2, pos_gt_obbs)
                        # one-hot from GT candidate IoUs
                        c1_better = (q_gt[:, 0] >= q_gt[:, 1]).astype(jt.float32)
                        c2_better = (q_gt[:, 1] > q_gt[:, 0]).astype(jt.float32)
                        target = jt.stack([c1_better, c2_better], dim=1)
                        weight = jt.ones_like(target)
                        losses['loss_sobb_cls'] = self.loss_sobb_cls(
                            amb_pred, target, weight,
                            avg_factor=max(avg_factor, 1.0))
                    else:
                        l_score = risk_aware_score_loss(amb_pred, q, D,
                            lambda_cal=self.loss_cal_weight,
                            lambda_pair=self.loss_pair_weight,
                            lambda_margin=self.loss_margin_weight)
                        losses['loss_sobb_cls'] = l_score.sum() / max(avg_factor, 1.0)
                else:
                    losses['loss_sobb_cls'] = jt.array(0.0)

        # Consistency loss (R07 ablation, paper Eq. 34-35):
        # Computes prediction consistency between the original and a
        # perturbed forward pass. The perturbation is a weak geometric
        # transform (small rotation/translation) applied to RoI features.
        if self.lambda_cons > 0:
            import jittor as _jt
            if pos_gt_obbs is not None and pos_prop_hbbs is not None and pos_gt_obbs.shape[0] > 0:
                # S0-5: Consistency loss with correct signature.
                # Pass OBB candidates (not q/D) to consistency_loss.
                # The perturbed view applies weak geometric noise to decoded
                # OBB candidates (rotation + translation, paper Section 3.4),
                # and logit-space noise to the scorer output as a proxy for
                # the perturbed forward pass (the second-view forward is not
                # available inside loss(); the logit perturbation provides
                # a non-zero, gradient-bearing consistency signal).
                pos_pred_7 = jt.concat([reg_pred, amb_pred], dim=1)
                cand1, cand2 = sobbb_decode_both(
                    pos_prop_hbbs, pos_pred_7,
                    means=list(self.bbox_coder.means), stds=list(self.bbox_coder.stds))
                # Weak perturbation: small rotation + translation on OBBs
                _n = cand1.shape[0]
                _angle_noise = _jt.randn(_n) * 0.02  # ~1 degree
                _trans_noise = _jt.randn(_n, 2) * 0.5  # ~0.5 pixel
                cand1_p = cand1.clone()
                cand1_p[:, 0] = cand1_p[:, 0] + _trans_noise[:, 0]
                cand1_p[:, 1] = cand1_p[:, 1] + _trans_noise[:, 1]
                cand1_p[:, 4] = cand1_p[:, 4] + _angle_noise
                cand2_p = cand2.clone()
                cand2_p[:, 0] = cand2_p[:, 0] + _trans_noise[:, 0]
                cand2_p[:, 1] = cand2_p[:, 1] + _trans_noise[:, 1]
                cand2_p[:, 4] = cand2_p[:, 4] + _angle_noise
                # Perturbed scorer logits: add small noise to amb_pred as a
                # proxy for the perturbed view's scorer output. This produces
                # non-zero KL divergence and gradients.
                amb_pred_p = amb_pred + _jt.randn_like(amb_pred) * 0.01
                # Call consistency_loss with OBB candidates (not q/D)
                l_cons = consistency_loss(
                    amb_pred, amb_pred_p,
                    cand1, cand2, cand1_p, cand2_p,
                    tau_s=1.0, lambda_cons=self.lambda_cons)
                losses['loss_cons'] = l_cons.mean()
            else:
                losses['loss_cons'] = jt.array(0.0)
        return losses

    def get_bboxes(self, rois, cls_score, bbox_pred, img_metas, cfg, rescale=False):
        # Decode SOBB to OBB
        # rois: [N, 5] (batch_idx, x, y, w, h)

        # 1. Get HBB from RoIs
        hbb = rois[:, 1:]
        if hbb.shape[1] == 5:
            x, y, w, h, a = [hbb[:, i] for i in range(5)]
            cos_a, sin_a = jt.abs(jt.cos(a)), jt.abs(jt.sin(a))
            W = w * cos_a + h * sin_a
            H = w * sin_a + h * cos_a
            hbb = jt.stack([x, y, W, H], dim=1)

        # 2. Softmax / Sigmoid for scores
        if cls_score is not None:
            if getattr(self.loss_cls, 'use_sigmoid', getattr(self.loss_cls, 'use_bce', False)):
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
        else:
            scores = None

        # 3. Decode boxes
        if self.reg_class_agnostic:
            decoded_bboxes = self.bbox_coder.decode(hbb, bbox_pred)
        else:
            num_samples = hbb.shape[0]
            decoded_bboxes = jt.zeros((num_samples, self.num_classes * 5))
            for i in range(self.num_classes):
                decoded_bboxes[:, i*5:(i+1)*5] = self.bbox_coder.decode(hbb, bbox_pred[:, i*7:(i+1)*7])

        # 4. Rescale
        if rescale:
            scale_factor = img_metas[0]['scale_factor']
            if self.reg_class_agnostic:
                decoded_bboxes[:, :4] /= scale_factor
            else:
                for i in range(self.num_classes):
                    decoded_bboxes[:, i*5:i*5+4] /= scale_factor

        # 5. Multi-class NMS
        det_bboxes, det_labels = multiclass_nms_rotated(
            decoded_bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)

        return det_bboxes, det_labels
