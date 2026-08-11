# Representation: Angle (standard 5-parameter angular OBB)
# Independent baseline: disables DSOF and SOBB scorer; uses standard
# DeltaXYWHABBoxCoder (5-param: dx,dy,dw,dh,da) and MaxIoUAssigner.
# Regression loss: SmoothL1 on the 5-parameter angle encoding.
_base_ = "../s2anet_r50_fpn_1x_rsdd.py"

model = dict(
    bbox_head=dict(
        loss_odm_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        train_cfg=dict(
            odm_cfg=dict(
                use_dsof=False,
                use_scorer=False,
                use_sala=False,
                repr_mode='angle',
                scorer_mode='none',
                lambda_cons=0.0,
                loss_cal_weight=0.0,
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
                    type='DeltaXYWHABBoxCoder',
                    target_means=(0., 0., 0., 0., 0.),
                    target_stds=(1., 1., 1., 1., 1.),
                    clip_border=True),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            )
        )
    )
)
