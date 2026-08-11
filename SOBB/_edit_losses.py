"""Apply all code fixes for the SOBB repo to match the revised paper.

Fixes:
1. Single-stage loss separation (s2anet_head.py): SmoothL1 for geometric params,
   BCE for candidate scores, sigmoid decoding for shape variable.
2. SALA S_pred passing (anchor_target.py): extract predicted S from shape head
   output and pass to SALAAssigner.
3. RoI Transformer inference path (roi_transformer.py): replace get_det_rbboxes
   with get_bboxes.
4. Add RSDD two-stage config.
"""
import os
import re

REPO = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(REPO, "python", "sobb")


def fix_s2anet_head():
    """Fix single-stage loss: separate geometric (SmoothL1) and scoring (BCE)."""
    fp = os.path.join(PY, "models", "roi_heads", "s2anet_head.py")
    with open(fp, "r", encoding="utf-8") as f:
        code = f.read()

    # 1. Add loss_odm_score parameter to __init__
    old_init = """             loss_odm_bbox=dict(
                 type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),"""
    new_init = """             loss_odm_bbox=dict(
                 type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
             loss_odm_score=dict(
                 type='CrossEntropyLoss', use_bce=True, loss_weight=1.0),"""
    assert old_init in code, "Could not find loss_odm_bbox in __init__ params"
    code = code.replace(old_init, new_init, 1)

    # 2. Build loss_odm_score in __init__ body
    old_build = """        self.loss_odm_bbox = build_from_cfg(loss_odm_bbox,LOSSES)"""
    new_build = """        self.loss_odm_bbox = build_from_cfg(loss_odm_bbox,LOSSES)
        self.loss_odm_score = build_from_cfg(loss_odm_score,LOSSES)"""
    assert old_build in code, "Could not find loss_odm_bbox build line"
    code = code.replace(old_build, new_build, 1)

    # 3. Replace loss_odm_single to split geometric and scoring losses
    old_loss = """    def loss_odm_single(self,
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
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 7)
        bbox_weights = bbox_weights.reshape(-1, 7)
        odm_bbox_pred = odm_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)

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
            odm_bbox_pred = bbox_coder.decode(anchors, odm_bbox_pred)
        loss_odm_bbox = self.loss_odm_bbox(
            odm_bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_odm_cls, loss_odm_bbox"""

    new_loss = """    def loss_odm_single(self,
                        odm_cls_score,
                        odm_bbox_pred,
                        anchors,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        num_total_samples,
                        cfg):
        # classification loss (FocalLoss)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        odm_cls_score = odm_cls_score.permute(0, 2, 3,
                                              1).reshape(-1, self.cls_out_channels)
        loss_odm_cls = self.loss_odm_cls(
            odm_cls_score, labels, label_weights, avg_factor=num_total_samples)

        # Split 7-param SOBB output into geometric regression and candidate
        # scoring (Eq. loss_function in the paper).
        #   reg_pred:   (dx, dy, dw, dh, z_s)  -- first 5 channels
        #   score_pred: (s1, s2)               -- last 2 channels
        # Targets from sobb_encode: (dx, dy, dw, dh, t_s, s1, s2)
        bbox_targets = bbox_targets.reshape(-1, 7)
        bbox_weights = bbox_weights.reshape(-1, 7)
        odm_bbox_pred = odm_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)

        reg_pred = odm_bbox_pred[:, :5]
        score_pred = odm_bbox_pred[:, 5:]
        reg_target = bbox_targets[:, :5]
        score_target = bbox_targets[:, 5:]
        reg_weight = bbox_weights[:, :5]
        score_weight = bbox_weights[:, 5:]

        # Decode shape logit z_s via sigmoid (Eq. ts_constraint)
        #   t_s_hat = clip(sigmoid(z_s), eps, 1-eps)
        # The regression loss is then SmoothL1 on (dx, dy, dw, dh, t_s_hat)
        # versus the target (dx_t, dy_t, dw_t, dh_t, t_s).
        eps = 1e-6
        z_s = reg_pred[:, 4]
        t_s_hat = jt.clamp(jt.sigmoid(z_s), eps, 1.0 - eps)
        reg_pred_decoded = jt.concat(
            [reg_pred[:, :4], t_s_hat.unsqueeze(1)], dim=1)

        loss_odm_bbox = self.loss_odm_bbox(
            reg_pred_decoded,
            reg_target,
            reg_weight,
            avg_factor=num_total_samples)

        # Candidate scoring loss: BCE on (s1, s2) with IoU soft targets
        # (Eq. scoring_loss in the paper)
        loss_odm_score = self.loss_odm_score(
            score_pred,
            score_target,
            score_weight,
            avg_factor=num_total_samples)

        return loss_odm_cls, loss_odm_bbox, loss_odm_score"""

    assert old_loss in code, "Could not find old loss_odm_single"
    code = code.replace(old_loss, new_loss, 1)

    # 4. Update loss() to handle 3 return values
    old_call = """        losses_odm_cls, losses_odm_bbox = multi_apply(
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

        return dict(loss_fam_cls=losses_fam_cls,
                    loss_fam_bbox=losses_fam_bbox,
                    loss_odm_cls=losses_odm_cls,
                    loss_odm_bbox=losses_odm_bbox)"""

    new_call = """        losses_odm_cls, losses_odm_bbox, losses_odm_score = multi_apply(
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

        return dict(loss_fam_cls=losses_fam_cls,
                    loss_fam_bbox=losses_fam_bbox,
                    loss_odm_cls=losses_odm_cls,
                    loss_odm_bbox=losses_odm_bbox,
                    loss_odm_score=losses_odm_score)"""

    assert old_call in code, "Could not find old multi_apply call in loss()"
    code = code.replace(old_call, new_call, 1)

    with open(fp, "w", encoding="utf-8") as f:
        f.write(code)
    print("  [OK] s2anet_head.py: separated SmoothL1 (geometric) and BCE (scoring)")


def fix_anchor_target_sala():
    """Pass predicted S_pred from the shape head to SALAAssigner.

    The paper (Sec. SALA) states that S_pred comes from the decoded candidate
    (the scale branch output t_s_hat, S_pred = 1 - t_s_hat).  The current
    anchor_target_single call does not pass pred_s, so SALAAssigner falls back
    to the geometric empty-area ratio of the anchor box.

    We add an optional pred_s_list parameter to anchor_target / anchor_target_single
    and, when the assigner accepts pred_s, pass it through.
    """
    fp = os.path.join(PY, "models", "boxes", "anchor_target.py")
    with open(fp, "r", encoding="utf-8") as f:
        code = f.read()

    # 1. Add pred_s_list parameter to anchor_target()
    old_sig = """def anchor_target(anchor_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True):"""
    new_sig = """def anchor_target(anchor_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True,
                  pred_s_list=None):"""
    assert old_sig in code, "Could not find anchor_target signature"
    code = code.replace(old_sig, new_sig, 1)

    # 2. Pass pred_s_list through multi_apply
    old_apply = """    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
        anchor_target_single,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        gt_bboxes_ignore_list,
        gt_labels_list,
        img_metas,
        target_means=target_means,
        target_stds=target_stds,
        cfg=cfg,
        label_channels=label_channels,
        sampling=sampling,
        unmap_outputs=unmap_outputs,
        num_level_anchors=num_level_anchors)"""
    new_apply = """    if pred_s_list is None:
        pred_s_list = [None] * num_imgs
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
        anchor_target_single,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        gt_bboxes_ignore_list,
        gt_labels_list,
        img_metas,
        target_means=target_means,
        target_stds=target_stds,
        cfg=cfg,
        label_channels=label_channels,
        sampling=sampling,
        unmap_outputs=unmap_outputs,
        num_level_anchors=num_level_anchors,
        pred_s_list=pred_s_list)"""
    assert old_apply in code, "Could not find multi_apply in anchor_target"
    code = code.replace(old_apply, new_apply, 1)

    # 3. Add pred_s parameter to anchor_target_single()
    old_single_sig = """def anchor_target_single(flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg=None,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True,
                         num_level_anchors=None):"""
    new_single_sig = """def anchor_target_single(flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg=None,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True,
                         num_level_anchors=None,
                         pred_s=None):"""
    assert old_single_sig in code, "Could not find anchor_target_single signature"
    code = code.replace(old_single_sig, new_single_sig, 1)

    # 4. Pass pred_s to assigner in both sampling and non-sampling branches
    # Sampling branch
    old_samp = """        _needs_lvl = 'num_level_bboxes' in inspect.signature(
            bbox_assigner.assign).parameters
        assign_args = [anchors]
        if _needs_lvl and num_level_anchors_inside is not None:
            assign_args.append(num_level_anchors_inside)
        assign_args.extend([gt_bboxes, gt_bboxes_ignore, None])

        assign_result = bbox_assigner.assign(*assign_args)
        sampling_result = bbox_sampler.sample(assign_result, anchors, gt_bboxes)"""
    new_samp = """        _needs_lvl = 'num_level_bboxes' in inspect.signature(
            bbox_assigner.assign).parameters
        _needs_pred_s = 'pred_s' in inspect.signature(
            bbox_assigner.assign).parameters
        assign_args = [anchors]
        if _needs_lvl and num_level_anchors_inside is not None:
            assign_args.append(num_level_anchors_inside)
        assign_args.extend([gt_bboxes, gt_bboxes_ignore, None])
        if _needs_pred_s and pred_s is not None:
            assign_args.append(pred_s)

        assign_result = bbox_assigner.assign(*assign_args)
        sampling_result = bbox_sampler.sample(assign_result, anchors, gt_bboxes)"""
    assert old_samp in code, "Could not find sampling branch in anchor_target_single"
    code = code.replace(old_samp, new_samp, 1)

    # Non-sampling branch
    old_nonsamp = """        _needs_lvl = 'num_level_bboxes' in inspect.signature(
            bbox_assigner.assign).parameters
        assign_args = [anchors]
        if _needs_lvl and num_level_anchors_inside is not None:
            assign_args.append(num_level_anchors_inside)
        assign_args.extend([gt_bboxes, gt_bboxes_ignore, gt_labels])

        assign_result = bbox_assigner.assign(*assign_args)
        bbox_sampler = PseudoSampler()"""
    new_nonsamp = """        _needs_lvl = 'num_level_bboxes' in inspect.signature(
            bbox_assigner.assign).parameters
        _needs_pred_s = 'pred_s' in inspect.signature(
            bbox_assigner.assign).parameters
        assign_args = [anchors]
        if _needs_lvl and num_level_anchors_inside is not None:
            assign_args.append(num_level_anchors_inside)
        assign_args.extend([gt_bboxes, gt_bboxes_ignore, gt_labels])
        if _needs_pred_s and pred_s is not None:
            assign_args.append(pred_s)

        assign_result = bbox_assigner.assign(*assign_args)
        bbox_sampler = PseudoSampler()"""
    assert old_nonsamp in code, "Could not find non-sampling branch"
    code = code.replace(old_nonsamp, new_nonsamp, 1)

    with open(fp, "w", encoding="utf-8") as f:
        f.write(code)
    print("  [OK] anchor_target.py: pred_s now passed to SALAAssigner")


def fix_s2anet_head_sala_pred_s():
    """Extract S_pred from odm_bbox_preds and pass to anchor_target for SALA.

    In the ODM stage, the 5th channel (index 4) of odm_bbox_pred is z_s (the
    shape logit).  We decode it to S_pred = 1 - clip(sigmoid(z_s), eps, 1-eps)
    and pass the per-image, level-concatenated S_pred to anchor_target.
    """
    fp = os.path.join(PY, "models", "roi_heads", "s2anet_head.py")
    with open(fp, "r", encoding="utf-8") as f:
        code = f.read()

    old_block = """        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            refine_anchors_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg.odm_cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)"""
    new_block = """        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        # Extract predicted S_pred from the ODM shape head for SALA.
        # odm_bbox_preds[i] has 7 channels: (dx, dy, dw, dh, z_s, s1, s2).
        # z_s is at index 4; S_pred = 1 - clip(sigmoid(z_s), eps, 1-eps).
        eps = 1e-6
        pred_s_list = []
        for img_id in range(len(img_metas)):
            per_img_s = []
            for lvl in range(len(odm_bbox_preds)):
                z_s = odm_bbox_preds[lvl][img_id, 4:5, :, :]  # [1, H, W]
                t_s_hat = jt.clamp(jt.sigmoid(z_s), eps, 1.0 - eps)
                s_pred = (1.0 - t_s_hat).squeeze(0)  # [H, W]
                s_pred = s_pred.reshape(-1)  # [H*W]
                per_img_s.append(s_pred)
            pred_s_list.append(jt.contrib.concat(per_img_s).detach())

        cls_reg_targets = anchor_target(
            refine_anchors_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg.odm_cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            pred_s_list=pred_s_list)"""
    assert old_block in code, "Could not find ODM anchor_target call in s2anet_head"
    code = code.replace(old_block, new_block, 1)

    with open(fp, "w", encoding="utf-8") as f:
        f.write(code)
    print("  [OK] s2anet_head.py: S_pred extracted from shape head and passed to SALA")


def fix_roi_transformer():
    """Replace get_det_rbboxes with get_bboxes in RoI Transformer test path."""
    fp = os.path.join(PY, "models", "networks", "roi_transformer.py")
    with open(fp, "r", encoding="utf-8") as f:
        code = f.read()

    old_call = """        rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats, rrois_enlarge)
        det_rbboxes, det_labels = self.rbbox_head.get_det_rbboxes(
            rrois,
            rcls_score,
            rbbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=self.test_cfg.rcnn)"""
    new_call = """        rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats, rrois_enlarge)
        det_rbboxes, det_labels = self.rbbox_head.get_bboxes(
            rrois,
            rcls_score,
            rbbox_pred,
            img_meta,
            self.test_cfg.rcnn,
            rescale=rescale)"""
    assert old_call in code, "Could not find get_det_rbboxes call"
    code = code.replace(old_call, new_call, 1)

    with open(fp, "w", encoding="utf-8") as f:
        f.write(code)
    print("  [OK] roi_transformer.py: get_det_rbboxes -> get_bboxes")


def add_rsdd_single_config():
    """Add RSDD single-stage S2ANet config based on the SSDD config."""
    cfg_dir = os.path.join(REPO, "projects", "s2anet", "configs")
    ssdd_cfg = os.path.join(cfg_dir, "s2anet_r50_fpn_1x_ssdd.py")
    rsdd_cfg = os.path.join(cfg_dir, "s2anet_r50_fpn_1x_rsdd.py")

    if not os.path.exists(ssdd_cfg):
        print("  [SKIP] SSDD config not found, cannot derive RSDD")
        return

    with open(ssdd_cfg, "r", encoding="utf-8") as f:
        code = f.read()

    code = code.replace("SSDDDataset", "RSDDDataset")
    code = code.replace("processed_SSDD", "processed_RSDD")
    code = code.replace("SSDD", "RSDD")

    with open(rsdd_cfg, "w", encoding="utf-8") as f:
        f.write(code)
    print("  [OK] Added s2anet_r50_fpn_1x_rsdd.py")


def add_two_stage_configs():
    """Add two-stage RoI Transformer configs for SSDD and RSDD.

    These configs wire up the RoITransformer model with:
      - RPN head (standard FPN-based)
      - bbox_head (standard HBB head)
      - rbbox_head = SOBBHead (the SOBB two-stage terminal head)
    """
    cfg_dir = os.path.join(REPO, "projects", "s2anet", "configs")

    template = '''# Two-stage RoI Transformer with SOBB terminal head
# Implements the Ours-Two model from the paper:
#   Stage-1: RPN + HBB RoI head (standard RoI Transformer)
#   Stage-2: Rotated RoI head replaced by SOBBHead (DSOF + SOBB + SALA)
#
# Dataset: {ds}
# Training: 120 epochs, batch_size=8, lr=0.005, NMS iou=0.1

model = dict(
    type='RoITransformer',
    backbone=dict(
        type='Resnet50',
        frozen_stages=1,
        return_stages=["layer1","layer2","layer3","layer4"],
        pretrained=True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_input",
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[4],
        anchor_ratios=[1.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        test_cfg=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='nms', iou_thr=0.7),
            max_per_img=1000),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlaps2D')),
            bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                            target_means=(0., 0., 0., 0.),
                            target_stds=(1., 1., 1., 1.)),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedHead',
        num_classes=2,
        in_channels=256,
        roi_feat_size=7,
        num_shared_fcs=2,
        fc_out_channels=1024,
        reg_class_agnostic=True,
        target_means=(0., 0., 0., 0.),
        target_stds=(1., 1., 1., 1.),
        loss_cls=dict(type='CrossEntropyLoss', use_bce=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    rbbox_roi_extractor=dict(
        type='RSingleRoIExtractor',
        roi_layer=dict(type='RRoIAlignRotated', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32],
        w_enlarge=1.2,
        h_enlarge=1.4),
    rbbox_head=dict(
        type='SOBBHead',
        num_classes=2,
        in_channels=256,
        roi_feat_size=7,
        num_shared_fcs=2,
        fc_out_channels=1024,
        reg_class_agnostic=True,
        target_means=(0., 0., 0., 0.),
        target_stds=(1., 1., 1., 1.),
        loss_cls=dict(type='CrossEntropyLoss', use_bce=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_sobb_cls=dict(type='CrossEntropyLoss', use_bce=True, loss_weight=1.0)),
    train_cfg=dict(
        rpn=dict(...),
        rpn_proposal=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='nms', iou_thr=0.7),
            max_per_img=1000),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D')),
                bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                                target_means=(0., 0., 0., 0.),
                                target_stds=(1., 1., 1., 1.)),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
                sampler=dict(type='RandomSampler')),
            dict(
                assigner=dict(
                    type='MaxIoUAssignerRbbox',
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D_rotated')),
                bbox_coder=dict(type='SOBBBBoxCoder',
                                target_means=(0., 0., 0., 0.),
                                target_stds=(1., 1., 1., 1.)),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
                sampler=dict(type='RandomSampler'))
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='nms', iou_thr=0.7),
            max_per_img=1000),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_thr=0.1),
            max_per_img=2000))
)

dataset = dict(
    train=dict(
        type="{ds}Dataset",
        dataset_dir='/path/to/your/processed_{ds}/train_800',
        transforms=[
            dict(type="RotatedResize", min_size=800, max_size=800),
            dict(type='RotatedRandomFlip', prob=0.5),
            dict(type="Pad", size_divisor=32),
            dict(type="Normalize",
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_bgr=False)
        ],
        batch_size=8,
        num_workers=4,
        shuffle=True,
        filter_empty_gt=False
    ),
    val=dict(
        type="{ds}Dataset",
        dataset_dir='/path/to/your/processed_{ds}/val_800',
        transforms=[
            dict(type="RotatedResize", min_size=800, max_size=800),
            dict(type="Pad", size_divisor=32),
            dict(type="Normalize",
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_bgr=False)
        ],
        batch_size=8,
        num_workers=4,
        shuffle=False
    )
)

optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001,
    grad_clip=dict(max_norm=35, norm_type=2))

scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    milestones=[80, 100])

logger = dict(type="RunLogger")

max_epoch = 120
eval_interval = 1
checkpoint_interval = 1
log_interval = 50
'''

    for ds in ["SSDD", "RSDD"]:
        out = os.path.join(cfg_dir, f"roitransformer_sobb_r50_fpn_1x_{ds.lower()}.py")
        with open(out, "w", encoding="utf-8") as f:
            f.write(template.format(ds=ds))
        print(f"  [OK] Added roitransformer_sobb_r50_fpn_1x_{ds.lower()}.py")


if __name__ == "__main__":
    print("=== Fixing single-stage loss (s2anet_head.py) ===")
    fix_s2anet_head()

    print("\n=== Fixing SALA pred_s passing (anchor_target.py) ===")
    fix_anchor_target_sala()

    print("\n=== Wiring S_pred from shape head to SALA (s2anet_head.py) ===")
    fix_s2anet_head_sala_pred_s()

    print("\n=== Fixing RoI Transformer inference path ===")
    fix_roi_transformer()

    print("\n=== Adding RSDD single-stage config ===")
    add_rsdd_single_config()

    print("\n=== Adding two-stage RoI Transformer configs ===")
    add_two_stage_configs()

    print("\n=== All code fixes applied ===")
