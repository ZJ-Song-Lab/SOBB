import jittor as jt
import inspect

from .sampler import PseudoSampler
from sobb.utils.general import multi_apply,unmap
from sobb.utils.registry import build_from_cfg,BOXES
from sobb.models.boxes.sobb_ops import sobb_encode


def assign_and_sample(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels, cfg):
    bbox_assigner = build_from_cfg(cfg.get('assigner', ''), BOXES)
    bbox_sampler = build_from_cfg(cfg.get('sampler', ''), BOXES)
    assign_result = bbox_assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore,
                                         gt_labels)
    sampling_result = bbox_sampler.sample(assign_result, bboxes, gt_bboxes,
                                          gt_labels)
    return assign_result, sampling_result


def anchor_target(anchor_list,
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
                  pred_s_list=None,
                  num_level_anchors=None,
                  bbox_coder_type='DeltaXYWH'):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    if num_level_anchors is None:
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = jt.contrib.concat(anchor_list[i])
        valid_flag_list[i] = jt.contrib.concat(valid_flag_list[i])

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    if pred_s_list is None:
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
        pred_s_list,
        target_means=target_means,
        target_stds=target_stds,
        cfg=cfg,
        label_channels=label_channels,
        sampling=sampling,
        unmap_outputs=unmap_outputs,
        num_level_anchors=num_level_anchors,
        bbox_coder_type=bbox_coder_type)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = jt.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


def anchor_target_single(flat_anchors,
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
                         num_level_anchors=None,
                         bbox_coder_type='DeltaXYWH'):
    bbox_coder_cfg = cfg.get('bbox_coder', '')
    if bbox_coder_cfg == '':
        bbox_coder_cfg = dict(type='DeltaXYWHBBoxCoder')
    bbox_coder = build_from_cfg(bbox_coder_cfg, BOXES)
    # Set True to use IoULoss
    reg_decoded_bbox = cfg.get('reg_decoded_bbox', False)

    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.get('allowed_border', -1))
    if not inside_flags.any(0):
        return (None,) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]
    pred_s_inside = pred_s[inside_flags] if pred_s is not None else None

    num_level_anchors_inside = None
    if num_level_anchors is not None:
        num_level_anchors_inside = []
        start = 0
        for n in num_level_anchors:
            end = start + n
            num_level_anchors_inside.append(inside_flags[start:end].sum().item())
            start = end

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
        if _needs_pred_s and pred_s_inside is not None:
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
        if _needs_pred_s and pred_s_inside is not None:
            assign_args.append(pred_s_inside)

        assign_result = bbox_assigner.assign(*assign_args)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    target_dim = 7 if bbox_coder_type == 'SOBB' else anchors.shape[1]
    bbox_targets = jt.zeros((num_valid_anchors, target_dim), dtype=anchors.dtype)
    bbox_weights = jt.zeros((num_valid_anchors, target_dim), dtype=anchors.dtype)
    labels = jt.zeros(num_valid_anchors).int()
    label_weights = jt.zeros(num_valid_anchors).float()
    # num_classes = 80
    # labels = jt.full((num_valid_anchors,), num_classes)
    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_bboxes = sampling_result.pos_bboxes
        pos_gt_bboxes = sampling_result.pos_gt_bboxes
        if bbox_coder_type == 'SOBB':
            # pos_bboxes are 5-D rotated anchors; compute outer HBB for
            # consistent reference geometry (matches inference decode).
            ax, ay, aw, ah, aa = (pos_bboxes[:, i] for i in range(5))
            cos_a = jt.abs(jt.cos(aa))
            sin_a = jt.abs(jt.sin(aa))
            pos_hbb = jt.stack([ax, ay, aw * cos_a + ah * sin_a,
                                 aw * sin_a + ah * cos_a], dim=1)
            pos_bbox_targets = sobb_encode(pos_hbb, pos_gt_bboxes,
                                           target_means, target_stds)
        elif not reg_decoded_bbox:
            pos_bbox_targets = bbox_coder.encode(pos_bboxes, pos_gt_bboxes)
        else:
            pos_bbox_targets = pos_gt_bboxes
        bbox_targets[pos_inds, :] = pos_bbox_targets.cast(bbox_targets.dtype)
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.get('pos_weight', -1)
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)


# TODO for rotated box
def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
                       (flat_anchors[:, 0] >= -allowed_border) & \
                       (flat_anchors[:, 1] >= -allowed_border) & \
                       (flat_anchors[:, 2] < img_w + allowed_border) & \
                       (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags



