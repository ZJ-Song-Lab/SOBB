import numpy as np
import jittor as jt
from jittor import nn

from sobb.models.utils.weight_init import normal_init,bias_init_with_prob
from sobb.models.utils.modules import ConvModule
from sobb.utils.general import multi_apply
from sobb.utils.registry import HEADS,LOSSES,BOXES,build_from_cfg


# from sobb.ops.dcn_v2 import DeformConv
from sobb.ops.dcn_v1 import DeformConv
from sobb.ops.orn import ORConv2d, RotationInvariantPooling
from sobb.ops.nms_rotated import multiclass_nms_rotated
from sobb.models.utils.dsof import DSOF, CandidateScorer
from sobb.models.boxes.box_ops import delta2bbox_rotated, rotated_box_to_poly
from sobb.models.boxes.sobb_ops import (sobb_candidate_geoms, dbbox_decode,
    sobbb_decode_both, reconstruct_gt_obb, compute_risk_aware_inputs, risk_aware_score_loss,
    consistency_loss)
from sobb.models.boxes.anchor_target import images_to_levels,anchor_target
from sobb.models.boxes.anchor_generator import AnchorGeneratorRotatedS2ANet


@HEADS.register_module()
class S2ANetHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=2,
                 with_orconv=True,
                 anchor_scales=[4],
                 anchor_ratios=[1.0],
                 anchor_strides=[8, 16, 32, 64, 128],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
                 loss_fam_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_fam_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_odm_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_odm_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_odm_sobb_cls=dict(
                     type='CrossEntropyLoss',
                     use_bce=True,
                     loss_weight=1.0),
                 test_cfg=dict(
                    nms_pre=2000,
                    min_bbox_size=0,
                    score_thr=0.05,
                    nms=dict(type='nms_rotated', iou_thr=0.1),
                    max_per_img=2000),
                train_cfg=dict(
                    fam_cfg=dict(
                        assigner=dict(
                            type='MaxIoUAssigner',
                            pos_iou_thr=0.5,
                            neg_iou_thr=0.4,
                            min_pos_iou=0,
                            ignore_iof_thr=-1,
                            iou_calculator=dict(type='BboxOverlaps2D_rotated')),
                        bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                                        target_means=(0., 0., 0., 0., 0.),
                                        target_stds=(1., 1., 1., 1., 1.),
                                        clip_border=True),
                        allowed_border=-1,
                        pos_weight=-1,
                        debug=False),
                    odm_cfg=dict(
                        assigner=dict(
                            type='SALAAssigner',
                            topk=9,
                            alpha=1.0,
                            beta=1.0,
                            iou_calculator=dict(type='BboxOverlaps2D_rotated')),
                        bbox_coder=dict(type='SOBBBBoxCoder',
                                        target_means=(0., 0., 0., 0.),
                                        target_stds=(1., 1., 1., 1.)),
                        allowed_border=-1,
                        pos_weight=-1,
                        debug=False))):
        super(S2ANetHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.with_orconv = with_orconv
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds

        self.use_sigmoid_cls = loss_odm_cls.get('use_sigmoid', False)
        self.sampling = loss_odm_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes

        if self.cls_out_channels <= 0:
            raise ValueError('num_classes={} is too small'.format(num_classes))
        self.loss_fam_cls = build_from_cfg(loss_fam_cls,LOSSES)
        self.loss_fam_bbox = build_from_cfg(loss_fam_bbox,LOSSES)
        self.loss_odm_cls = build_from_cfg(loss_odm_cls,LOSSES)
        self.loss_odm_bbox = build_from_cfg(loss_odm_bbox,LOSSES)
        self.loss_odm_sobb_cls = build_from_cfg(loss_odm_sobb_cls,LOSSES)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # ODM-specific 4-element means/stds for SOBB 7-parameter decoding
        _odm_coder = train_cfg.get('odm_cfg', {}).get('bbox_coder', {})
        self.odm_target_means = _odm_coder.get('target_means', (0., 0., 0., 0.))
        self.odm_target_stds = _odm_coder.get('target_stds', (1., 1., 1., 1.))
        self.lambda_cons = train_cfg.get('lambda_cons', 0.0)
        # Scorer training mode: 'scorer_only' (default, detach features+candidates),
        # 'feature_joint' (detach candidates only, feature gets gradient),
        # 'fully_joint' (no detach, full end-to-end gradient)
        self.scorer_mode = train_cfg.get('scorer_mode', 'scorer_only')
        self.loss_cal_weight = train_cfg.get('loss_cal_weight', 1.0)
        self.loss_pair_weight = train_cfg.get('loss_pair_weight', 1.0)
        self.loss_margin_weight = train_cfg.get('loss_margin_weight', 1.0)
        self.use_dsof = train_cfg.get('use_dsof', True)
        self.use_scorer = train_cfg.get('use_scorer', True)
        self.use_sala = train_cfg.get('use_sala', True)
        self.consistency_mode = train_cfg.get('consistency_mode', 'logit_noise')
        self.repr_mode = train_cfg.get('repr_mode', 'sobb')

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(AnchorGeneratorRotatedS2ANet(anchor_base, anchor_scales, anchor_ratios))

        # anchor cache
        self.base_anchors = dict()
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU()
        self.fam_reg_convs = nn.ModuleList()
        self.fam_cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.fam_reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            self.fam_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))

        self.fam_reg = nn.Conv2d(self.feat_channels, 5, 1)
        self.fam_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)

        self.align_conv = AlignConv(
            self.feat_channels, self.feat_channels, kernel_size=3)

        if self.with_orconv:
            self.or_conv = ORConv2d(self.feat_channels, int(
                self.feat_channels / 8), kernel_size=3, padding=1, arf_config=(1, 8))
        else:
            self.or_conv = nn.Conv2d(
                self.feat_channels, self.feat_channels, 3, padding=1)
        self.or_pool = RotationInvariantPooling(256, 8)

        self.odm_reg_convs = nn.ModuleList()
        self.odm_cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = int(self.feat_channels /
                      8) if i == 0 and self.with_orconv else self.feat_channels
            self.odm_reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            self.odm_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))

        self.odm_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        
        # C1: Conditional DSOF and scorer
        if self.use_dsof:
            self.dsof = DSOF(self.feat_channels)
        else:
            self.dsof = None
        if self.use_scorer:
            self.candidate_scorer = CandidateScorer(self.feat_channels)
        else:
            self.candidate_scorer = None
        if self.use_scorer:
            self.odm_reg_scale = nn.Conv2d(self.feat_channels, 3, 3, padding=1)
            self.odm_reg_align = nn.Conv2d(self.feat_channels, 2, 3, padding=1)
        else:
            self.odm_reg_scale = nn.Conv2d(self.feat_channels, 2, 3, padding=1)
            self.odm_reg_align = nn.Conv2d(self.feat_channels, 3, 3, padding=1)

        self.init_weights()

    def init_weights(self):
        for m in self.fam_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.fam_cls_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fam_reg, std=0.01)
        normal_init(self.fam_cls, std=0.01, bias=bias_cls)

        self.align_conv.init_weights()

        normal_init(self.or_conv, std=0.01)
        for m in self.odm_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.odm_cls_convs:
            normal_init(m.conv, std=0.01)
        normal_init(self.odm_cls, std=0.01, bias=bias_cls)
        normal_init(self.odm_reg_scale, std=0.01)
        normal_init(self.odm_reg_align, std=0.01)

    def forward_single(self, x, stride):
        fam_reg_feat = x
        for fam_reg_conv in self.fam_reg_convs:
            fam_reg_feat = fam_reg_conv(fam_reg_feat)
        fam_bbox_pred = self.fam_reg(fam_reg_feat)

        # only forward during training
        if self.is_training():
            fam_cls_feat = x
            for fam_cls_conv in self.fam_cls_convs:
                fam_cls_feat = fam_cls_conv(fam_cls_feat)
            fam_cls_score = self.fam_cls(fam_cls_feat)
        else:
            fam_cls_score = None

        num_level = self.anchor_strides.index(stride)
        featmap_size = tuple(fam_bbox_pred.shape[-2:])
        if (num_level, featmap_size) in self.base_anchors:
            init_anchors = self.base_anchors[(num_level, featmap_size)]
        else:
            init_anchors = self.anchor_generators[num_level].grid_anchors(featmap_size, self.anchor_strides[num_level])
            self.base_anchors[(num_level, featmap_size)] = init_anchors

        refine_anchor = bbox_decode(
            fam_bbox_pred.detach(),
            init_anchors,
            self.target_means,
            self.target_stds)

        align_feat = self.align_conv(x, refine_anchor.clone(), stride)

        or_feat = self.or_conv(align_feat)
        odm_reg_feat = or_feat
        if self.with_orconv:
            odm_cls_feat = self.or_pool(or_feat)
        else:
            odm_cls_feat = or_feat

        for odm_reg_conv in self.odm_reg_convs:
            odm_reg_feat = odm_reg_conv(odm_reg_feat)
        for odm_cls_conv in self.odm_cls_convs:
            odm_cls_feat = odm_cls_conv(odm_cls_feat)
        odm_cls_score = self.odm_cls(odm_cls_feat)
        
        # C1: Conditional DSOF
        if self.use_dsof:
            f_scale, f_align = self.dsof(odm_reg_feat)
        else:
            f_scale = odm_reg_feat
            f_align = odm_reg_feat
        reg_scale = self.odm_reg_scale(f_scale)
        reg_align = self.odm_reg_align(f_align)

        # Decode predicted HBB dimensions and S for candidate geometry.
        # refine_anchor: [N, H, W, 5] = (x, y, w, h, angle)
        ra_w = refine_anchor[..., 2]
        ra_h = refine_anchor[..., 3]
        ra_a = refine_anchor[..., 4]
        cos_a = jt.abs(jt.cos(ra_a))
        sin_a = jt.abs(jt.sin(ra_a))
        hbb_w = ra_w * cos_a + ra_h * sin_a
        hbb_h = ra_w * sin_a + ra_h * cos_a

        dw = reg_scale[:, 0]
        dh = reg_scale[:, 1]
        ds = reg_scale[:, 2]
        gw = jt.exp(dw) * hbb_w
        gh = jt.exp(dh) * hbb_h
        eps = 1e-6
        t_s = jt.clamp(jt.sigmoid(ds), eps, 1.0 - eps)
        S = 1.0 - t_s

        x1, y1, x2, y2 = sobb_candidate_geoms(gw, gh, S)
        gw_safe = jt.maximum(gw, eps)
        gh_safe = jt.maximum(gh, eps)
        log_ratio = jt.log(gw_safe / gh_safe)
        cand1 = jt.stack([x1 / gw_safe, y1 / gh_safe, log_ratio, S], dim=-1)
        cand2 = jt.stack([x2 / gw_safe, y2 / gh_safe, log_ratio, S], dim=-1)
        candidates = jt.stack([cand1, cand2], dim=1)  # [N, 2, H, W, 4]

        # C1: Conditional scorer / output assembly
        if self.use_scorer:
            if self.scorer_mode == 'fully_joint':
                scorer_feat = f_align
                scorer_cands = candidates
            elif self.scorer_mode == 'feature_joint':
                scorer_feat = f_align
                scorer_cands = candidates.detach()
            else:
                scorer_feat = f_align.detach()
                scorer_cands = candidates.detach()
            s_scores = self.candidate_scorer(scorer_feat, scorer_cands)
            # SOBB 7-param: [dx, dy, dw, dh, ds, s1, s2]
            odm_bbox_pred = jt.concat([
                reg_align[:, 0:1, :, :], reg_align[:, 1:2, :, :],
                reg_scale[:, 0:1, :, :], reg_scale[:, 1:2, :, :], reg_scale[:, 2:3, :, :],
                s_scores[:, 0:1, :, :], s_scores[:, 1:2, :, :]
            ], dim=1)
        else:
            # Standard 5-param OBB: [dx, dy, dw, dh, da]
            odm_bbox_pred = jt.concat([
                reg_align[:, 0:1, :, :], reg_align[:, 1:2, :, :],
                reg_scale[:, 0:1, :, :], reg_scale[:, 1:2, :, :],
                reg_align[:, 2:3, :, :]
            ], dim=1)

        return fam_cls_score, fam_bbox_pred, refine_anchor, odm_cls_score, odm_bbox_pred

    def get_init_anchors(self,
                         featmap_sizes,
                         img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                w,h = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list

    def get_refine_anchors(self,
                           featmap_sizes,
                           refine_anchors,
                           img_metas,
                           is_train=True):
        num_levels = len(featmap_sizes)

        refine_anchors_list = []
        for img_id, img_meta in enumerate(img_metas):
            mlvl_refine_anchors = []
            for i in range(num_levels):
                refine_anchor = refine_anchors[i][img_id].reshape(-1, 5)
                mlvl_refine_anchors.append(refine_anchor)
            refine_anchors_list.append(mlvl_refine_anchors)

        valid_flag_list = []
        if is_train:
            for img_id, img_meta in enumerate(img_metas):
                multi_level_flags = []
                for i in range(num_levels):
                    anchor_stride = self.anchor_strides[i]
                    feat_h, feat_w = featmap_sizes[i]
                    w,h = img_meta['pad_shape'][:2]
                    valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                    valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                    flags = self.anchor_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w))
                    multi_level_flags.append(flags)
                valid_flag_list.append(multi_level_flags)
        return refine_anchors_list, valid_flag_list

    def loss(self,
             fam_cls_scores,
             fam_bbox_preds,
             refine_anchors,
             odm_cls_scores,
             odm_bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        
        cfg = self.train_cfg.copy()
        featmap_sizes = [featmap.size()[-2:] for featmap in odm_cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_init_anchors(featmap_sizes, img_metas)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(jt.contrib.concat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,num_level_anchors)

        # Feature Alignment Module
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg.fam_cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,num_total_pos, num_total_neg = cls_reg_targets
        
        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos

        losses_fam_cls, losses_fam_bbox = multi_apply(
            self.loss_fam_single,
            fam_cls_scores,
            fam_bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg.fam_cfg)

        # Oriented Detection Module targets
        refine_anchors_list, valid_flag_list = self.get_refine_anchors(
            featmap_sizes, refine_anchors, img_metas)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0)
                             for anchors in refine_anchors_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(refine_anchors_list)):
            concat_anchor_list.append(jt.contrib.concat(refine_anchors_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        # Extract predicted S (Empty Area Ratio) from ODM bbox predictions for SALA.
        # z_s is the 5th channel (index 4); S_pred = 1 - sigmoid(z_s), detached.
        num_imgs = len(img_metas)
        num_levels = len(odm_bbox_preds)
        pred_s_list = []
        for img_id in range(num_imgs):
            z_s_per_level = []
            for lvl in range(num_levels):
                z_s = odm_bbox_preds[lvl][img_id, 4, :, :]
                z_s_per_level.append(z_s.reshape(-1))
            z_s_flat = jt.concat(z_s_per_level)
            t_s = jt.sigmoid(z_s_flat)
            S_pred = (1.0 - t_s).detach()
            pred_s_list.append(S_pred)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        odm_coder_type = 'SOBB' if cfg.odm_cfg.get('bbox_coder', {}).get('type', '') == 'SOBBBBoxCoder' else 'DeltaXYWH'
        cls_reg_targets = anchor_target(
            refine_anchors_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.odm_target_means,
            self.odm_target_stds,
            cfg.odm_cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            pred_s_list=pred_s_list,
            num_level_anchors=num_level_anchors,
            bbox_coder_type=odm_coder_type)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        losses_odm_cls, losses_odm_bbox, losses_odm_sobb_cls = multi_apply(
            self.loss_odm_single,
            odm_cls_scores,
            odm_bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg.odm_cfg)

        losses = dict(loss_fam_cls=losses_fam_cls,
                    loss_fam_bbox=losses_fam_bbox,
                    loss_odm_cls=losses_odm_cls,
                    loss_odm_bbox=losses_odm_bbox,
                    loss_odm_sobb_cls=losses_odm_sobb_cls)
        # S0-5: Consistency loss for single-stage path.
        # Apply weak geometric noise to decoded OBB candidates and
        # logit-space noise to scorer logits as a proxy for the perturbed
        # forward pass. This produces non-zero, gradient-bearing consistency.
        if self.lambda_cons > 0:
            import jittor as _jt
            _cons_losses = []
            for _lvl_idx in range(len(odm_bbox_preds)):
                _bbox_pred = odm_bbox_preds[_lvl_idx]
                _labels = labels_list[_lvl_idx]
                _anchors = all_anchor_list[_lvl_idx]
                _reg_pred = _bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                _labels_flat = _labels.reshape(-1)
                _pos_inds = (_labels_flat > 0)
                if _pos_inds.any():
                    _pos_reg = _reg_pred[_pos_inds]
                    _pos_anchors = _anchors.reshape(-1, 5)[_pos_inds]
                    _pos_amb = _pos_reg[:, 5:]
                    # Convert OBB anchors to HBB for sobbb_decode_both
                    _ax = _pos_anchors[:, 0]
                    _ay = _pos_anchors[:, 1]
                    _aw = _pos_anchors[:, 2]
                    _ah = _pos_anchors[:, 3]
                    _aa = _pos_anchors[:, 4]
                    _cos_a = jt.abs(jt.cos(_aa))
                    _sin_a = jt.abs(jt.sin(_aa))
                    _pos_anchor_hbb = jt.stack([
                        _ax, _ay,
                        _aw * _cos_a + _ah * _sin_a,
                        _aw * _sin_a + _ah * _cos_a], dim=1)
                    _c1, _c2 = sobbb_decode_both(
                        _pos_anchor_hbb, _pos_reg,
                        means=list(self.odm_target_means),
                        stds=list(self.odm_target_stds))
                    _n = _c1.shape[0]
                    _ang_n = _jt.randn(_n) * 0.02
                    _tr_n = _jt.randn(_n, 2) * 0.5
                    _c1p = _c1.clone()
                    _c1p[:, 0] = _c1p[:, 0] + _tr_n[:, 0]
                    _c1p[:, 1] = _c1p[:, 1] + _tr_n[:, 1]
                    _c1p[:, 4] = _c1p[:, 4] + _ang_n
                    _c2p = _c2.clone()
                    _c2p[:, 0] = _c2p[:, 0] + _tr_n[:, 0]
                    _c2p[:, 1] = _c2p[:, 1] + _tr_n[:, 1]
                    _c2p[:, 4] = _c2p[:, 4] + _ang_n
                    # Perturbed logits: add noise for non-zero KL
                    _amb_p = _pos_amb + _jt.randn_like(_pos_amb) * 0.01
                    _l = consistency_loss(
                        _pos_amb, _amb_p,
                        _c1, _c2, _c1p, _c2p,
                        tau_s=1.0, lambda_cons=self.lambda_cons)
                    _cons_losses.append(_l.mean())
                else:
                    _cons_losses.append(_jt.array(0.0))
            losses['loss_cons'] = _cons_losses
        return losses

    def loss_fam_single(self,
                        fam_cls_score,
                        fam_bbox_pred,
                        anchors,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        num_total_samples,
                        cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        fam_cls_score = fam_cls_score.permute(
            0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_fam_cls = self.loss_fam_cls(
            fam_cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        fam_bbox_pred = fam_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)

        reg_decoded_bbox = cfg.get('reg_decoded_bbox', False)
        if reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_coder_cfg = cfg.get('bbox_coder', '')
            if bbox_coder_cfg == '':
                bbox_coder_cfg = dict(type='DeltaXYWHBBoxCoder')
            bbox_coder = build_from_cfg(bbox_coder_cfg,BOXES)
            anchors = anchors.reshape(-1, 5)
            fam_bbox_pred = bbox_coder.decode(anchors, fam_bbox_pred)
        loss_fam_bbox = self.loss_fam_bbox(
            fam_bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_fam_cls, loss_fam_bbox

    def loss_odm_single(self,
                        odm_cls_score,
                        odm_bbox_pred,
                        anchors,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        num_total_samples,
                        cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        odm_cls_score = odm_cls_score.permute(0, 2, 3,
                                              1).reshape(-1, self.cls_out_channels)
        loss_odm_cls = self.loss_odm_cls(
            odm_cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss: split geometric (SmoothL1) and candidate scores (BCE)
        bbox_targets = bbox_targets.reshape(-1, 7)
        bbox_weights = bbox_weights.reshape(-1, 7)
        odm_bbox_pred = odm_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)

        # Split: reg_pred = [dx, dy, dw, dh, z_s] (first 5), amb_pred = [s1, s2] (last 2)
        reg_pred = odm_bbox_pred[:, :5]
        amb_pred = odm_bbox_pred[:, 5:]
        reg_target = bbox_targets[:, :5]
        amb_target = bbox_targets[:, 5:]

        # Decode-then-loss for the shape parameter t_s:
        # t_s_hat = clip(sigmoid(z_s), eps, 1-eps)
        eps = 1e-6
        z_s = reg_pred[:, 4]
        t_s_hat = jt.clamp(jt.sigmoid(z_s), eps, 1.0 - eps)
        reg_pred_decoded = jt.concat([reg_pred[:, :4], t_s_hat.unsqueeze(1)], dim=1)

        loss_odm_bbox = self.loss_odm_bbox(
            reg_pred_decoded,
            reg_target,
            bbox_weights[:, :5],
            avg_factor=num_total_samples)

        # Risk-aware pairwise scoring loss (calibration + pairwise + margin)
        anchors = anchors.reshape(-1, 5)
        pos_inds = (labels > 0)
        if pos_inds.any():
            pos_anchors = anchors[pos_inds]
            ax, ay, aw, ah, aa = (pos_anchors[:, i] for i in range(5))
            cos_a = jt.abs(jt.cos(aa))
            sin_a = jt.abs(jt.sin(aa))
            pos_anchor_hbb = jt.stack([
                ax, ay, aw * cos_a + ah * sin_a, aw * sin_a + ah * cos_a], dim=1)
            pos_reg_pred = reg_pred[pos_inds]
            pos_amb_pred = amb_pred[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_pred_7 = jt.concat([pos_reg_pred, pos_amb_pred], dim=1)
            cand1, cand2 = sobbb_decode_both(
                pos_anchor_hbb, pos_pred_7,
                means=list(self.odm_target_means), stds=list(self.odm_target_stds))
            gt_obb = reconstruct_gt_obb(
                pos_anchor_hbb, pos_bbox_targets,
                means=list(self.odm_target_means), stds=list(self.odm_target_stds))
            q, D = compute_risk_aware_inputs(cand1, cand2, gt_obb)
            l_score = risk_aware_score_loss(pos_amb_pred, q, D,
                lambda_cal=self.loss_cal_weight,
                lambda_pair=self.loss_pair_weight,
                lambda_margin=self.loss_margin_weight)
            loss_odm_sobb_cls = l_score.sum() / num_total_samples
        else:
            loss_odm_sobb_cls = amb_pred.sum() * 0.0
        return loss_odm_cls, loss_odm_bbox, loss_odm_sobb_cls

    def get_bboxes(self,
                   fam_cls_scores,
                   fam_bbox_preds,
                   refine_anchors,
                   odm_cls_scores,
                   odm_bbox_preds,
                   img_metas,
                   rescale=True):
        assert len(odm_cls_scores) == len(odm_bbox_preds)
        cfg = self.test_cfg.copy()

        featmap_sizes = [featmap.size()[-2:] for featmap in odm_cls_scores]
        num_levels = len(odm_cls_scores)

        refine_anchors = self.get_refine_anchors(
            featmap_sizes, refine_anchors, img_metas, is_train=False)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                odm_cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                odm_bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               refine_anchors[0][img_id], img_shape,
                                               scale_factor, cfg, rescale)

            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_score_list,
                          bbox_pred_list,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.
        """
        
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(
                1, 2, 0).reshape(-1, self.cls_out_channels)

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
            # anchors = rect2rbox(anchors)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores = scores.max(dim=1)
                else:
                    max_scores = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            
            # Use dbbox_decode for SOBB 7-parameter decoding to [x,y,w,h,a]
            bboxes = dbbox_decode(anchors, bbox_pred, self.odm_target_means, self.odm_target_stds)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = jt.contrib.concat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes[..., :4] /= scale_factor
        mlvl_scores = jt.contrib.concat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = jt.zeros((mlvl_scores.shape[0], 1),dtype=mlvl_scores.dtype)
            mlvl_scores = jt.contrib.concat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms_rotated(mlvl_bboxes,
                                                        mlvl_scores,
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)
        boxes = det_bboxes[:, :5]
        scores = det_bboxes[:, 5]
        polys = rotated_box_to_poly(boxes)
        return polys, scores, det_labels

    
    def parse_targets(self,targets,is_train=True):
        img_metas = []
        gt_bboxes = []
        gt_bboxes_ignore = []
        gt_labels = []

        for target in targets:
            if is_train:
                gt_bboxes.append(target["rboxes"])
                gt_labels.append(target["labels"])
                gt_bboxes_ignore.append(target["rboxes_ignore"])
            img_metas.append(dict(
                img_shape=target["img_size"][::-1],
                scale_factor=target["scale_factor"],
                pad_shape = target["pad_shape"]
            ))
        if not is_train:
            return img_metas
        return gt_bboxes,gt_labels,img_metas,gt_bboxes_ignore

    def execute(self, feats,targets):
        outs = multi_apply(self.forward_single, feats, self.anchor_strides)
        if self.is_training():
            return self.loss(*outs,*self.parse_targets(targets))
        else:
            return self.get_bboxes(*outs,self.parse_targets(targets,is_train=False))

def bbox_decode(
        bbox_preds,
        anchors,
        means=[0, 0, 0, 0, 0],
        stds=[1, 1, 1, 1, 1]):
    """
    Decode bboxes from deltas
    :param bbox_preds: [N,5,H,W]
    :param anchors: [H*W,5]
    :param means: mean value to decode bbox
    :param stds: std value to decode bbox
    :return: [N,H,W,5]
    """
    num_imgs, _, H, W = bbox_preds.shape
    bboxes_list = []
    for img_id in range(num_imgs):
        bbox_pred = bbox_preds[img_id]
        # bbox_pred.shape=[5,H,W]
        bbox_delta = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
        bboxes = delta2bbox_rotated(
            anchors, bbox_delta, means, stds, wh_ratio_clip=1e-6)
        bboxes = bboxes.reshape(H, W, 5)
        bboxes_list.append(bboxes)
    return jt.stack(bboxes_list, dim=0)


class AlignConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=1):
        super(AlignConv, self).__init__()
        self.kernel_size = kernel_size
        self.deform_conv = DeformConv(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      padding=(kernel_size - 1) // 2,
                                      deformable_groups=deformable_groups)
        self.relu = nn.ReLU()

    def init_weights(self):
        normal_init(self.deform_conv, std=0.01)

    @jt.no_grad()
    def get_offset(self, anchors, featmap_size, stride):
        dtype = anchors.dtype
        feat_h, feat_w = featmap_size
        pad = (self.kernel_size - 1) // 2
        idx = jt.arange(-pad, pad + 1, dtype=dtype)
        yy, xx = jt.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # get sampling locations of default conv
        xc = jt.arange(0, feat_w, dtype=dtype)
        yc = jt.arange(0, feat_h, dtype=dtype)
        yc, xc = jt.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        x_conv = xc[:, None] + xx
        y_conv = yc[:, None] + yy

        # get sampling locations of anchors
        x_ctr, y_ctr, w, h, a = jt.unbind(anchors, dim=1)
        x_ctr, y_ctr, w, h = x_ctr / stride, y_ctr / stride, w / stride, h / stride
        cos, sin = jt.cos(a), jt.sin(a)
        dw, dh = w / self.kernel_size, h / self.kernel_size
        x, y = dw[:, None] * xx, dh[:, None] * yy
        xr = cos[:, None] * x - sin[:, None] * y
        yr = sin[:, None] * x + cos[:, None] * y
        x_anchor, y_anchor = xr + x_ctr[:, None], yr + y_ctr[:, None]
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = jt.stack([offset_y, offset_x], dim=-1)
        # NA,ks*ks*2
        offset = offset.reshape(anchors.size(
            0), -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        return offset

    def execute(self, x, anchors, stride):
        num_imgs, H, W = anchors.shape[:3]
        offset_list = [
            self.get_offset(anchors[i].reshape(-1, 5), (H, W), stride)
            for i in range(num_imgs)
        ]
        offset_tensor = jt.stack(offset_list, dim=0)
        x = self.relu(self.deform_conv(x, offset_tensor))
        return x
