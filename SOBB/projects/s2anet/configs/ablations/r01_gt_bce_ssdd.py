# Series R ablation: GT-candidate independent BCE (no risk-aware scoring, Series R row 1)
#
# C2 fix: This config now disables DSOF (use_dsof=False) and replaces the
# SALA assigner with a standard MaxIoUAssigner (use_sala=False), isolating
# the GT-candidate source from the feature-decoupling and adaptive-label
# assignment used in R02+.  The loss() path in SOBBHead detects that all
# three risk loss weights are zero and constructs candidates from the
# GT-encoded target (not from predictions), providing a true GT-candidate
# set as defined in the paper.
#
# Usage: python tools/run_net.py --config-file projects/s2anet/configs/ablations/r01_gt_bce_ssdd.py --seed 1
#   Use --seed {1,2,3,4,5} for the five-seed protocol.

_base_ = "../roitransformer_r50_fpn_1x_ssdd.py"

model = dict(
    rbbox_head=dict(
        use_dsof=False,
        use_scorer=True,
        use_sala=False,
        consistency_mode='logit_noise',
        repr_mode='sobb',
        scorer_mode='scorer_only',
        lambda_cons=0.0,
        loss_cal_weight=0.0,
        loss_pair_weight=0.0,
        loss_margin_weight=0.0,
    ),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D')),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                bbox_coder=dict(
                    type='DeltaXYWHRBBoxCoder',
                    target_means=(0., 0., 0., 0., 0.),
                    target_stds=(1., 1., 1., 1., 1.)),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D_rotated')),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                bbox_coder=dict(
                    type='SOBBBBoxCoder',
                    target_means=(0., 0., 0., 0.),
                    target_stds=(1., 1., 1., 1.)),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
        ]
    )
)
