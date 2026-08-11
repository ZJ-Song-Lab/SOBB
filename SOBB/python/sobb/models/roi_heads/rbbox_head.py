import jittor as jt
from jittor import nn

from sobb.ops.bbox_transforms import (dbbox2delta_v3, delta2dbbox_v3,
                                      hbb2obb_v2, choose_best_Rroi_batch)
from sobb.ops.nms_rotated import multiclass_nms_rotated
from sobb.utils.registry import HEADS, LOSSES, build_from_cfg
from sobb.models.utils.weight_init import normal_init


@HEADS.register_module()
class BBoxHeadRbbox(nn.Module):
    """Base class for the first-stage bbox head in RoI Transformer.

    The first-stage head receives horizontal RoI features from the RPN and
    predicts 5-parameter deltas (dx, dy, dw, dh, dangle) to refine the
    horizontal RoIs into rotated proposals.  The second-stage head
    (``SOBBHead``) then consumes these rotated proposals and predicts the
    final SOBB 7-parameter output.

    The encoding/decoding uses ``dbbox2delta_v3`` / ``delta2dbbox_v3`` so that
    the 5-parameter representation is (x, y, w, h, angle) in rotated format.
    """

    def __init__(self,
                 num_classes=81,
                 in_channels=2048,
                 roi_feat_size=7,
                 with_avg_pool=True,
                 with_cls=True,
                 with_reg=True,
                 reg_class_agnostic=False,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.),
                 loss_cls=dict(type='CrossEntropyLoss',
                               use_sigmoid=False,
                               loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1Loss',
                                beta=1.0 / 9.0,
                                loss_weight=1.0)):
        super(BBoxHeadRbbox, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.roi_feat_size = roi_feat_size
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.reg_class_agnostic = reg_class_agnostic
        self.target_means = target_means
        self.target_stds = target_stds

        self.loss_cls = build_from_cfg(loss_cls, LOSSES)
        self.loss_bbox = build_from_cfg(loss_bbox, LOSSES)

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        if self.with_cls:
            self.fc_cls = nn.Linear(self.in_channels, self.cls_out_channels)
        if self.with_reg:
            out_dim_reg = 5 if self.reg_class_agnostic else 5 * self.num_classes
            self.fc_reg = nn.Linear(self.in_channels, out_dim_reg)

    def init_weights(self):
        if self.with_cls:
            normal_init(self.fc_cls, std=0.01)
        if self.with_reg:
            normal_init(self.fc_reg, std=0.001)

    def execute(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target(self, sampling_results, gt_obbs, gt_labels,
                   rcnn_train_cfg):
        """Encode 5-parameter regression targets.

        The first-stage sampler operates on horizontal boxes, so the positive
        proposals are HBBs.  They are converted to OBBs via ``hbb2obb_v2``
        (angle = -pi/2) before encoding 5-parameter deltas against the
        assigned OBB ground truth.

        Args:
            sampling_results (list[SamplingResult]): sampling results from
                the first-stage HBB assigner/sampler.
            gt_obbs (list[Tensor]): ground truth OBBs per image, each
                [num_gt, 5] = (x, y, w, h, angle).
            gt_labels (list[Tensor]): ground truth labels per image.
            rcnn_train_cfg (dict): training config for the first stage.

        Returns:
            tuple: (labels, label_weights, bbox_targets, bbox_weights)
                - labels: [N] int class labels (0 = background).
                - label_weights: [N] float weights.
                - bbox_targets: [N, 5] encoded deltas.
                - bbox_weights: [N, 5] weights (1 for positives).
        """
        labels_list = []
        label_weights_list = []
        bbox_targets_list = []
        bbox_weights_list = []

        for i, res in enumerate(sampling_results):
            num_samples = res.bboxes.shape[0]
            labels = jt.zeros((num_samples,), dtype=jt.int32)
            label_weights = jt.zeros((num_samples,), dtype=jt.float32)
            bbox_targets = jt.zeros((num_samples, 5), dtype=jt.float32)
            bbox_weights = jt.zeros((num_samples, 5), dtype=jt.float32)

            num_pos = res.pos_bboxes.shape[0]
            if num_pos > 0:
                pos_proposals_hbb = res.pos_bboxes
                pos_proposals_obb = hbb2obb_v2(pos_proposals_hbb)

                gt_obb = gt_obbs[i]
                if gt_obb is not None and gt_obb.numel() > 0:
                    pos_gt_inds = res.pos_assigned_gt_inds
                    pos_gt_obbs = gt_obb[pos_gt_inds.long()]
                else:
                    pos_gt_obbs = jt.zeros((num_pos, 5), dtype=jt.float32)

                pos_bbox_targets = dbbox2delta_v3(
                    pos_proposals_obb, pos_gt_obbs,
                    self.target_means, self.target_stds)

                bbox_targets[:num_pos, :] = pos_bbox_targets
                bbox_weights[:num_pos, :] = 1.0

                pos_labels = res.pos_gt_labels
                if pos_labels is not None and pos_labels.numel() > 0:
                    labels[:num_pos] = pos_labels
                    label_weights[:num_pos] = 1.0

            labels_list.append(labels)
            label_weights_list.append(label_weights)
            bbox_targets_list.append(bbox_targets)
            bbox_weights_list.append(bbox_weights)

        labels = jt.concat(labels_list, dim=0)
        label_weights = jt.concat(label_weights_list, dim=0)
        bbox_targets = jt.concat(bbox_targets_list, dim=0)
        bbox_weights = jt.concat(bbox_weights_list, dim=0)

        return labels, label_weights, bbox_targets, bbox_weights

    def loss(self, cls_score, bbox_pred, labels, label_weights,
             bbox_targets, bbox_weights):
        """Compute classification + 5-parameter regression loss."""
        losses = dict()
        if cls_score is not None:
            avg_factor = jt.sum(label_weights > 0).float().item()
            avg_factor = max(avg_factor, 1.0)
            losses['loss_cls'] = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=avg_factor)

        if bbox_pred is not None:
            pos_inds = (labels > 0)
            if pos_inds.any():
                pos_bbox_pred = bbox_pred[pos_inds]
                pos_labels = labels[pos_inds]

                if not self.reg_class_agnostic:
                    pos_bbox_pred = pos_bbox_pred.view(
                        -1, self.num_classes, 5)
                    idx = jt.arange(pos_bbox_pred.shape[0])
                    pos_bbox_pred = pos_bbox_pred[idx, pos_labels]

                pos_bbox_targets = bbox_targets[pos_inds]
                pos_bbox_weights = bbox_weights[pos_inds]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred, pos_bbox_targets, pos_bbox_weights,
                    avg_factor=avg_factor)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0

        return losses

    @jt.no_grad()
    def refine_rbboxes(self, rois, labels, bbox_pred, pos_is_gts,
                       img_metas):
        """Refine horizontal RoIs into rotated proposals during training.

        Args:
            rois (Tensor): [N, 6] = (batch_ind, x, y, w, h, angle) from
                ``roi2droi`` (HBB converted to OBB with angle = -pi/2).
            labels (Tensor): [N] class labels (from ``get_target``).
            bbox_pred (Tensor): [N, 5] or [N, 5*num_classes] predicted deltas.
            pos_is_gts (list[Tensor]): per-image bool flags marking GT
                proposals among positives.
            img_metas (list[dict]): image meta info.

        Returns:
            list[Tensor]: refined OBB proposals per image, each [N_i, 5] =
            (x, y, w, h, angle).
        """
        if not self.reg_class_agnostic:
            N = rois.shape[0]
            bbox_pred = bbox_pred.view(N, self.num_classes, 5)
            idx = jt.arange(N)
            bbox_pred = bbox_pred[idx, labels]

        refined = delta2dbbox_v3(
            rois[:, 1:], bbox_pred,
            self.target_means, self.target_stds)

        proposals = []
        num_imgs = len(img_metas)
        for img_id in range(num_imgs):
            inds = (rois[:, 0] == img_id)
            img_refined = refined[inds]
            proposals.append(img_refined)

        return proposals

    @jt.no_grad()
    def regress_by_class_rbbox(self, rois, label, bbox_pred, img_meta):
        """Regress rotated RoIs by class during testing.

        Args:
            rois (Tensor): [N, 6] = (batch_ind, x, y, w, h, angle) from
                ``roi2droi``.
            label (Tensor): [N] predicted class labels.
            bbox_pred (Tensor): [N, 5] or [N, 5*num_classes] predicted deltas.
            img_meta (dict): single image meta.

        Returns:
            Tensor: [N, 6] = (batch_ind, x, y, w, h, angle) refined OBB RoIs.
        """
        N = rois.shape[0]
        if not self.reg_class_agnostic:
            bbox_pred = bbox_pred.view(N, self.num_classes, 5)
            idx = jt.arange(N)
            bbox_pred = bbox_pred[idx, label]

        refined = delta2dbbox_v3(
            rois[:, 1:], bbox_pred,
            self.target_means, self.target_stds)

        batch_ind = rois[:, 0:1]
        rrois = jt.contrib.concat([batch_ind, refined], dim=1)
        return rrois
