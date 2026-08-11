import jittor as jt
import numpy as np
import jittor.nn as nn
from sobb.utils.registry import BOXES, MODELS, build_from_cfg, BACKBONES, HEADS, NECKS, ROI_EXTRACTORS
from sobb.ops.bbox_transforms import bbox2roi, gt_mask_bp_obbs_list, roi2droi, choose_best_Rroi_batch, dbbox2roi, dbbox2result
import copy

@MODELS.register_module()
class RoITransformer(nn.Module):
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 rbbox_roi_extractor=None,
                 rbbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RoITransformer, self).__init__()

        self.backbone = build_from_cfg(backbone, BACKBONES)
        self.neck = build_from_cfg(neck, NECKS)
        self.rpn_head = build_from_cfg(rpn_head, HEADS)
        self.bbox_roi_extractor = build_from_cfg(bbox_roi_extractor, ROI_EXTRACTORS)
        self.bbox_head = build_from_cfg(bbox_head, HEADS)
        self.rbbox_roi_extractor = build_from_cfg(rbbox_roi_extractor, ROI_EXTRACTORS)
        self.rbbox_head = build_from_cfg(rbbox_head, HEADS)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # S0-10: S_pred comes from the joint regression branch (t_w, t_h, t_s)
        # of the first-stage SOBBHead, NOT from a separate auxiliary MLP.
        # The bbox_head already predicts z_s as the 5th parameter of its
        # 7-element output (dx, dy, dw, dh, z_s, s1, s2). We extract z_s
        # from cls_score, bbox_pred = self.bbox_head(bbox_feats) and compute
        # S_pred = 1 - sigmoid(z_s) for use in SALA assignment.
        # No separate proposal_scale_predictor is needed.

    def execute_train(self, images, targets=None):
        self.backbone.train()

        image_meta = []
        gt_labels = []
        gt_bboxes = []
        gt_bboxes_ignore = []
        gt_obbs = []
        for target in targets:
            meta = dict(
                ori_shape = target['ori_img_size'],
                img_shape = target['img_size'],
                pad_shape = target['pad_shape'],
                img_file = target['img_file'],
                to_bgr = target['to_bgr'],
                scale_factor = target['scale_factor']
            )
            image_meta.append(meta)
            gt_bboxes.append(target['hboxes'])
            gt_labels.append(target['labels'])
            gt_bboxes_ignore.append(target['hboxes_ignore'])
            gt_obbs.append(target['rboxes'])

        losses = dict()
        features = self.backbone(images)
        if(self.neck):
            features = self.neck(features)
        rpn_outs = self.rpn_head(features)
        rpn_loss_inputs = rpn_outs + (gt_bboxes, image_meta,
                                        self.train_cfg.rpn)
        rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(rpn_losses)

        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                            self.test_cfg.rpn)
        proposal_inputs = rpn_outs + (image_meta, proposal_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

        bbox_assigner = build_from_cfg(self.train_cfg.rcnn[0].assigner, BOXES)
        bbox_sampler = build_from_cfg(self.train_cfg.rcnn[0].sampler, BOXES) #ingnored: context=self
        num_imgs = images.shape[0]
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []

        for proposal, gt_bbox, gt_bbox_ignore, gt_label in zip(proposal_list, gt_bboxes, gt_bboxes_ignore, gt_labels):
            assign_result = bbox_assigner.assign(
                proposal[:,:4], gt_bbox, gt_bbox_ignore, gt_label
            )
            sampling_result = bbox_sampler.sample(
                assign_result, proposal, gt_bbox, gt_label  
            )
            sampling_results.append(sampling_result)

        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(
            features[:self.bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        rbbox_targets = self.bbox_head.get_target(
            sampling_results, gt_obbs, gt_labels, self.train_cfg.rcnn[0])
        loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, *rbbox_targets)
        for name, value in loss_bbox.items():
            losses['s{}.{}'.format(0, name)] = (value)

        #RoITransfomer
        pos_is_gts = [res.pos_is_gt for res in sampling_results]
        roi_labels = rbbox_targets[0]

        with jt.no_grad():
            rotated_proposal_list = self.bbox_head.refine_rbboxes(
                roi2droi(rois), roi_labels, bbox_pred, pos_is_gts, image_meta
            )
        
        # assign gts and sample proposals (rbb assign)
        bbox_assigner = build_from_cfg(self.train_cfg.rcnn[1].assigner, BOXES)
        bbox_sampler = build_from_cfg(self.train_cfg.rcnn[1].sampler, BOXES)
        num_imgs = images.shape[0]
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]

        # S0-10: Extract S_pred from the first-stage SOBBHead's joint
        # regression output. bbox_pred[:, 4] is z_s (the 5th parameter),
        # which is part of the (t_w, t_h, t_s) joint prediction branch.
        # S_pred = 1 - sigmoid(z_s) provides a gradient-flowing learned
        # scale for SALA proposal assignment.
        if bbox_pred.shape[0] > 0:
            # bbox_pred is [N, 7] (dx, dy, dw, dh, z_s, s1, s2)
            # when reg_class_agnostic, or [N, 7*num_classes] otherwise.
            # For the agnostic case, z_s is column 4.
            if bbox_pred.shape[1] >= 7:
                z_s_proposal = bbox_pred[:, 4]
            else:
                z_s_proposal = jt.zeros(bbox_pred.shape[0])
            _t_s = jt.clamp(jt.sigmoid(z_s_proposal), 1e-6, 1.0 - 1e-6)
            pred_s_all = (1.0 - _t_s).detach()
        else:
            pred_s_all = None
            z_s_proposal = None

        # Track per-image proposal offsets for indexing pred_s_all
        _prop_counts = [res.bboxes.shape[0] for res in sampling_results]
        _prop_offsets = [0]
        for _c in _prop_counts:
            _prop_offsets.append(_prop_offsets[-1] + _c)

        # Save first-stage sampling results for scale loss computation;
        # sampling_results will be reassigned below for second-stage.
        _rpn_sampling_results = sampling_results

        sampling_results = []
        for _img_idx, (rotated_proposal, gt_obb, gt_bbox_ignore, gt_label) in enumerate(zip(rotated_proposal_list, gt_obbs, gt_bboxes_ignore, gt_labels)):
            gt_obbs_best_roi = jt.array(choose_best_Rroi_batch(gt_obb))
            num_level_bboxes = [rotated_proposal.shape[0]]
            # Learned S_pred (paper Eq. 8): 1 - sigmoid(z_s)
            if rotated_proposal.shape[0] > 0 and pred_s_all is not None:
                _start = _prop_offsets[_img_idx]
                _end = _prop_offsets[_img_idx + 1]
                pred_s = pred_s_all[_start:_end]
            else:
                pred_s = None
            assign_result = bbox_assigner.assign(
                rotated_proposal, num_level_bboxes, gt_obbs_best_roi,
                gt_bbox_ignore, gt_label, pred_s=pred_s
            )
            sampling_result = bbox_sampler.sample(
                assign_result, rotated_proposal, gt_obbs_best_roi, gt_label
            )
            sampling_results.append(sampling_result)

        # S0-10: Scale prediction loss using z_s from the joint regression
        # branch. z_s_proposal comes from bbox_pred[:, 4], which is the
        # t_s parameter jointly predicted with (t_w, t_h) in the first-stage
        # SOBBHead. This provides a gradient signal to the regression branch
        # to learn S_pred, matching the paper's description of S_pred coming
        # from the joint (t_w, t_h, t_s) prediction.
        if z_s_proposal is not None and z_s_proposal.shape[0] > 0:
            _pos_mask = jt.zeros((z_s_proposal.shape[0],), dtype=jt.bool)
            _pos_idx = 0
            for _res in _rpn_sampling_results:
                _n_pos = _res.pos_bboxes.shape[0]
                _n_total = _res.bboxes.shape[0]
                if _n_pos > 0:
                    _pos_mask[_pos_idx:_pos_idx + _n_pos] = True
                _pos_idx += _n_total
            if _pos_mask.any():
                _z_s_pos = z_s_proposal[_pos_mask]
                # Target: geometric S of the matched GT OBB.
                # pos_gt_bboxes is HBB (4 cols), so we index gt_obbs via
                # pos_assigned_gt_inds to get OBB (x, y, w, h, angle).
                _all_pos_gt_list = []
                for _img_idx, _res in enumerate(_rpn_sampling_results):
                    if _res.pos_assigned_gt_inds.numel() > 0:
                        _gt_obbs_img = gt_obbs[_img_idx]
                        _all_pos_gt_list.append(
                            _gt_obbs_img[_res.pos_assigned_gt_inds, :])
                if len(_all_pos_gt_list) > 0:
                    _all_pos_gt = jt.concat(_all_pos_gt_list, dim=0)
                else:
                    _all_pos_gt = None
                if _all_pos_gt is not None and _all_pos_gt.shape[0] > 0:
                    _pw = _all_pos_gt[:, 2]
                    _ph = _all_pos_gt[:, 3]
                    _pa = _all_pos_gt[:, 4]
                    _cos = jt.abs(jt.cos(_pa))
                    _sin = jt.abs(jt.sin(_pa))
                    _ow = _pw * _cos + _ph * _sin
                    _oh = _pw * _sin + _ph * _cos
                    _oa = jt.maximum(_ow * _oh, jt.array(1e-8))
                    _obb_a = _pw * _ph
                    _s_gt = jt.clamp(1.0 - _obb_a / _oa, 0.0, 1.0 - 1e-6).detach()
                    _t_gt = (1.0 - _s_gt).clamp(1e-6, 1.0 - 1e-6)
                    _z_gt = jt.log(_t_gt / (1.0 - _t_gt))
                    losses['loss_proposal_scale'] = jt.smooth_l1_loss(
                        _z_s_pos, _z_gt, reduction='mean') * 0.1

        # (batch_ind, x_ctr, y_ctr, w, h, angle)
        rrois = dbbox2roi([res.bboxes for res in sampling_results])

        # feat enlarge
        # rrois[:, 3] = rrois[:, 3] * 1.2
        # rrois[:, 4] = rrois[:, 4] * 1.4
        rrois[:, 3] = rrois[:, 3] * self.rbbox_roi_extractor.w_enlarge
        rrois[:, 4] = rrois[:, 4] * self.rbbox_roi_extractor.h_enlarge
        rbbox_feats = self.rbbox_roi_extractor(features[:self.rbbox_roi_extractor.num_inputs], rrois)
        cls_score, rbbox_pred = self.rbbox_head(rbbox_feats, rrois)
        rbbox_targets = self.rbbox_head.get_target_rbbox(
            sampling_results, gt_obbs, gt_labels, self.train_cfg.rcnn[1])
        loss_rbbox = self.rbbox_head.loss(cls_score, rbbox_pred, *rbbox_targets)
        for name, value in loss_rbbox.items():
            losses['s{}.{}'.format(1, name)] = (value)
        return losses

    def execute_test(self, images, targets=None, rescale=False):
        '''
        Args:
            images (jt.Var): image tensors, shape is [N,C,H,W]
            targets (list[dict]): targets for each image
        Rets:
            losses (dict): losses
        '''
        img_meta = []
        img_shape = []
        scale_factor = []
        for target in targets:
            ori_img_size = target['ori_img_size']
            meta = dict(
                ori_shape = ori_img_size,
                img_shape = ori_img_size,
                pad_shape = ori_img_size,
                scale_factor = target['scale_factor'],
                img_file = target['img_file']
            )
            img_meta.append(meta)
            img_shape.append(target['img_size'])
            scale_factor.append(target['scale_factor'])
        x = self.backbone(images)
        if(self.neck):
            x = self.neck(x)

        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

        rois = bbox2roi(proposal_list)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = self.bbox_head(roi_feats)

        bbox_label = jt.argmax(cls_score, dim=1)[0]
        rrois = self.bbox_head.regress_by_class_rbbox(roi2droi(rois), bbox_label, bbox_pred,
                                                      img_meta[0])
        rrois_enlarge = copy.deepcopy(rrois)
        rrois_enlarge[:, 3] = rrois_enlarge[:, 3] * self.rbbox_roi_extractor.w_enlarge
        rrois_enlarge[:, 4] = rrois_enlarge[:, 4] * self.rbbox_roi_extractor.h_enlarge
        rbbox_feats = self.rbbox_roi_extractor(
            x[:len(self.rbbox_roi_extractor.featmap_strides)], rrois_enlarge)
        rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats, rrois_enlarge)
        det_rbboxes, det_labels = self.rbbox_head.get_bboxes(
            rrois,
            rcls_score,
            rbbox_pred,
            img_meta,
            self.test_cfg.rcnn,
            rescale=rescale)

        rbbox_results = dbbox2result(det_rbboxes, det_labels,
                                     self.rbbox_head.num_classes)
        return [rbbox_results]
    def execute(self, images, targets=None):
        if self.is_training():
            return self.execute_train(images, targets)
        else:
            return self.execute_test(images, targets)
