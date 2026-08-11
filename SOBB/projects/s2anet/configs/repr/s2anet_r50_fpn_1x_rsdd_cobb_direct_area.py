# Representation: COBB-direct-area (two-candidate, direct-area r_a)
# C5 baseline: uses the direct-area variable r_a (not the sliding ratio
# r_s) to produce exactly two candidate OBBs, then scores them with the
# same shared CandidateScorer as SOBB.  This isolates the contribution
# of the closed-form solver and shared scoring from the 4-candidate
# sliding-ratio COBB baseline.
#
# In COBB (CVPR 2024), the direct-area r_a directly parametrises the
# convex hull area split.  The same (x_c, y_c, w, h, r_a) tuple yields
# exactly two symmetric OBB solutions.  SOBB's t_s = r_a, so this
# baseline shares the SOBB two-candidate geometry but uses COBB's
# direct-area parametrisation and lacks the risk-aware margin.
_base_ = "../s2anet_r50_fpn_1x_rsdd.py"

model = dict(
    bbox_head=dict(
        train_cfg=dict(
            odm_cfg=dict(
                use_dsof=False,
                use_scorer=True,
                use_sala=False,
                repr_mode='cobb_direct_area',
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
