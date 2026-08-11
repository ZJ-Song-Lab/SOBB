"""
Phase 4 edits: Fix single-stage loss, SALA S_pred, RoI Transformer inference.
"""
import re

HEAD = r"C:\Users\tesst\Desktop\SOBB_repo\SOBB\python\sobb\models\roi_heads\s2anet_head.py"
AT   = r"C:\Users\tesst\Desktop\SOBB_repo\SOBB\python\sobb\models\boxes\anchor_target.py"
RT   = r"C:\Users\tesst\Desktop\SOBB_repo\SOBB\python\sobb\models\networks\roi_transformer.py"

# ============================================================
# Edit 1: s2anet_head.py - Add loss_odm_sobb_cls, fix loss_odm_single
# ============================================================
with open(HEAD, 'r', encoding='utf-8') as f:
    src = f.read()

# 1a. Add loss_odm_sobb_cls config after loss_odm_bbox
old = '''                 loss_odm_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 test_cfg=dict('''
new = '''                 loss_odm_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_odm_sobb_cls=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 test_cfg=dict('''
assert old in src, "1a not found"
src = src.replace(old, new)

# 1b. Build loss_odm_sobb_cls after loss_odm_bbox build
old = "        self.loss_odm_bbox = build_from_cfg(loss_odm_bbox,LOSSES)"
new = "        self.loss_odm_bbox = build_from_cfg(loss_odm_bbox,LOSSES)\n        self.loss_odm_sobb_cls = build_from_cfg(loss_odm_sobb_cls,LOSSES)"
assert old in src, "1b not found"
src = src.replace(old, new)

# 1c. Replace loss_odm_single with split-loss version
old_loss = '''    def loss_odm_single(self,
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
        return loss_odm_cls, loss_odm_bbox'''

new_loss = '''    def loss_odm_single(self,
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
        # Split 7-param SOBB into geometric (dx,dy,dw,dh,z_s) and
        # candidate scores (s1, s2). Per paper: geometric uses SmoothL1
        # with sigmoid-decoded z_s; candidate scores use BCE.
        bbox_targets = bbox_targets.reshape(-1, 7)
        bbox_weights = bbox_weights.reshape(-1, 7)
        odm_bbox_pred = odm_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)

        reg_pred = odm_bbox_pred[:, :5]
        amb_pred = odm_bbox_pred[:, 5:]
        reg_target = bbox_targets[:, :5]
        amb_target = bbox_targets[:, 5:]

        eps = 1e-6
        z_s = reg_pred[:, 4]
        t_s_hat = jt.clamp(jt.sigmoid(z_s), eps, 1.0 - eps)
        reg_pred_decoded = jt.concat(
            [reg_pred[:, :4], t_s_hat.unsqueeze(1)], dim=1)

        loss_odm_bbox = self.loss_odm_bbox(
            reg_pred_decoded,
            reg_target,
            bbox_weights[:, :5],
            avg_factor=num_total_samples)

        loss_odm_sobb_cls = self.loss_odm_sobb_cls(
            amb_pred,
            amb_target,
            bbox_weights[:, 5:],
            avg_factor=num_total_samples)
        return loss_odm_cls, loss_odm_bbox, loss_odm_sobb_cls'''

assert old_loss in src, "1c not found"
src = src.replace(old_loss, new_loss)

# 1d. Update multi_apply call and return dict in loss()
old_call = '''        losses_odm_cls, losses_odm_bbox = multi_apply(
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
                    loss_odm_bbox=losses_odm_bbox)'''

new_call = '''        losses_odm_cls, losses_odm_bbox, losses_odm_sobb_cls = multi_apply(
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
                    loss_odm_sobb_cls=losses_odm_sobb_cls)'''

assert old_call in src, "1d not found"
src = src.replace(old_call, new_call)

# 3. Add S_pred extraction before ODM anchor_target call
old_odm = '''        # Oriented Detection Module targets
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

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
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
            sampling=self.sampling)'''

new_odm = '''        # Oriented Detection Module targets
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

        # Extract terminal S_pred from SOBB shape head output (z_s channel).
        # S_pred = 1 - clip(sigmoid(z_s), eps, 1-eps), detached for assignment.
        num_imgs = len(img_metas)
        pred_s_list = []
        for img_id in range(num_imgs):
            per_img_s = []
            for level_id in range(len(odm_bbox_preds)):
                z_s = odm_bbox_preds[level_id][img_id, 4, :, :]
                t_s_hat = jt.clamp(jt.sigmoid(z_s), 1e-6, 1.0 - 1e-6)
                s_pred = (1.0 - t_s_hat).detach()
                per_img_s.append(s_pred.reshape(-1))
            pred_s_list.append(jt.contrib.concat(per_img_s))

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
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
            pred_s_list=pred_s_list)'''

assert old_odm in src, "3 not found"
src = src.replace(old_odm, new_odm)

with open(HEAD, 'w', encoding='utf-8') as f:
    f.write(src)
print("Edit 1+3 (s2anet_head.py): DONE")

# ============================================================
# Edit 2: anchor_target.py - Add pred_s support for SALA
# ============================================================
with open(AT, 'r', encoding='utf-8') as f:
    src = f.read()

# 2a. Add pred_s_list parameter to anchor_target function
old_at_sig = '''def anchor_target(anchor_list,
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
                  unmap_outputs=True):'''
new_at_sig = '''def anchor_target(anchor_list,
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
                  pred_s_list=None):'''
assert old_at_sig in src, "2a not found"
src = src.replace(old_at_sig, new_at_sig)

# 2b. Pass pred_s_list to multi_apply call
old_multi = '''    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
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
        num_level_anchors=num_level_anchors)'''
new_multi = '''    # Build per-image pred_s list for SALA (None entries when not provided)
    if pred_s_list is None:
        pred_s_list_per_img = [None] * len(img_metas)
    else:
        pred_s_list_per_img = pred_s_list
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
        anchor_target_single,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        gt_bboxes_ignore_list,
        gt_labels_list,
        img_metas,
        pred_s_list_per_img,
        target_means=target_means,
        target_stds=target_stds,
        cfg=cfg,
        label_channels=label_channels,
        sampling=sampling,
        unmap_outputs=unmap_outputs,
        num_level_anchors=num_level_anchors)'''
assert old_multi in src, "2b not found"
src = src.replace(old_multi, new_multi)

# 2c. Add pred_s parameter to anchor_target_single signature
old_single_sig = '''def anchor_target_single(flat_anchors,
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
                         num_level_anchors=None):'''
new_single_sig = '''def anchor_target_single(flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         pred_s,
                         target_means,
                         target_stds,
                         cfg=None,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True,
                         num_level_anchors=None):'''
assert old_single_sig in src, "2c not found"
src = src.replace(old_single_sig, new_single_sig)

# 2d. Filter pred_s by inside_flags and pass to assigner
old_sample = '''    if sampling:
        bbox_assigner = build_from_cfg(cfg.get('assigner', ''), BOXES)
        bbox_sampler = build_from_cfg(cfg.get('sampler', ''), BOXES)

        _needs_lvl = 'num_level_bboxes' in inspect.signature(
            bbox_assigner.assign).parameters
        assign_args = [anchors]
        if _needs_lvl and num_level_anchors_inside is not None:
            assign_args.append(num_level_anchors_inside)
        assign_args.extend([gt_bboxes, gt_bboxes_ignore, None])

        assign_result = bbox_assigner.assign(*assign_args)
        sampling_result = bbox_sampler.sample(assign_result, anchors, gt_bboxes)
    else:
        bbox_assigner = build_from_cfg(cfg.get('assigner', ''), BOXES)

        _needs_lvl = 'num_level_bboxes' in inspect.signature(
            bbox_assigner.assign).parameters
        assign_args = [anchors]
        if _needs_lvl and num_level_anchors_inside is not None:
            assign_args.append(num_level_anchors_inside)
        assign_args.extend([gt_bboxes, gt_bboxes_ignore, gt_labels])

        assign_result = bbox_assigner.assign(*assign_args)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)'''

new_sample = '''    # Filter pred_s to match inside anchors (for SALA terminal S_pred)
    pred_s_inside = None
    if pred_s is not None:
        pred_s_inside = pred_s[inside_flags]

    if sampling:
        bbox_assigner = build_from_cfg(cfg.get('assigner', ''), BOXES)
        bbox_sampler = build_from_cfg(cfg.get('sampler', ''), BOXES)

        _needs_lvl = 'num_level_bboxes' in inspect.signature(
            bbox_assigner.assign).parameters
        _needs_pred_s = 'pred_s' in inspect.signature(
            bbox_assigner.assign).parameters
        assign_args = [anchors]
        if _needs_lvl and num_level_anchors_inside is not None:
            assign_args.append(num_level_anchors_inside)
        assign_args.extend([gt_bboxes, gt_bboxes_ignore, None])
        if _needs_pred_s:
            assign_args.append(pred_s_inside)

        assign_result = bbox_assigner.assign(*assign_args)
        sampling_result = bbox_sampler.sample(assign_result, anchors, gt_bboxes)
    else:
        bbox_assigner = build_from_cfg(cfg.get('assigner', ''), BOXES)

        _needs_lvl = 'num_level_bboxes' in inspect.signature(
            bbox_assigner.assign).parameters
        _needs_pred_s = 'pred_s' in inspect.signature(
            bbox_assigner.assign).parameters
        assign_args = [anchors]
        if _needs_lvl and num_level_anchors_inside is not None:
            assign_args.append(num_level_anchors_inside)
        assign_args.extend([gt_bboxes, gt_bboxes_ignore, gt_labels])
        if _needs_pred_s:
            assign_args.append(pred_s_inside)

        assign_result = bbox_assigner.assign(*assign_args)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)'''

assert old_sample in src, "2d not found"
src = src.replace(old_sample, new_sample)

with open(AT, 'w', encoding='utf-8') as f:
    f.write(src)
print("Edit 2 (anchor_target.py): DONE")

# ============================================================
# Edit 4: roi_transformer.py - Fix get_det_rbboxes -> get_bboxes
# ============================================================
with open(RT, 'r', encoding='utf-8') as f:
    src = f.read()

old_roi = '''        rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats, rrois_enlarge)
        det_rbboxes, det_labels = self.rbbox_head.get_det_rbboxes(
            rrois,
            rcls_score,
            rbbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=self.test_cfg.rcnn)

        rbbox_results = dbbox2result(det_rbboxes, det_labels,
                                     self.rbbox_head.num_classes)
        return [rbbox_results]'''

new_roi = '''        rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats, rrois_enlarge)
        # SOBBHead provides get_bboxes (not get_det_rbboxes); use the
        # paper's inference interface for the two-stage decoding path.
        img_metas_single = [dict(
            img_shape=img_shape[0],
            scale_factor=scale_factor[0])]
        det_rbboxes, det_labels = self.rbbox_head.get_bboxes(
            rrois,
            rcls_score,
            rbbox_pred,
            img_metas_single,
            self.test_cfg.rcnn,
            rescale=rescale)

        rbbox_results = dbbox2result(det_rbboxes, det_labels,
                                     self.rbbox_head.num_classes)
        return [rbbox_results]'''

assert old_roi in src, "4 not found"
src = src.replace(old_roi, new_roi)

with open(RT, 'w', encoding='utf-8') as f:
    f.write(src)
print("Edit 4 (roi_transformer.py): DONE")

print("\nAll Phase 4 edits completed successfully.")
