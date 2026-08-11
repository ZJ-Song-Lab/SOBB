# Representation: COBB (Convex OBB, four-candidate sliding-ratio)
# Independent baseline: disables DSOF; uses SOBB scorer infrastructure
# in 4-candidate mode (repr_mode='cobb'). The head forward path calls
# cobb_ops.cobb_decode_both to produce four candidate OBBs from the
# predicted (dx, dy, dw, dh, z_s) + four logits, then scores them with
# the shared CandidateScorer.  Uses calibration-only scoring (no
# pairwise or margin, since COBB has its own 4-candidate geometry).
_base_ = "../s2anet_r50_fpn_1x_rsdd.py"

model = dict(
    bbox_head=dict(
        train_cfg=dict(
            odm_cfg=dict(
                use_dsof=False,
                use_scorer=True,
                use_sala=False,
                repr_mode='cobb',
                scorer_mode='scorer_only',
                lambda_cons=0.0,
                loss_cal_weight=1.0,
                loss_pair_weight=0.0,
                loss_margin_weight=0.0,
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D_rotated')),
                bbox_coder=dict(
                    type='SOBBBBoxCoder',
                    target_means=(0., 0., 0., 0.),
                    target_stds=(1., 1., 1., 1.)),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            )
        )
    )
)
