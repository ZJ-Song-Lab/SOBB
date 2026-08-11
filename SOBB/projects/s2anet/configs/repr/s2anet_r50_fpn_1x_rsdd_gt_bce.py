# Representation: GT-candidate BCE SOBB (single-stage, RSDD)
# Table T row: GT-candidate independent BCE on the single-stage S2ANet.
# All three risk loss weights are zero, so the loss path in
# S2ANetHead constructs candidates from the GT-encoded target and
# applies independent BCE scoring (no pairwise, no margin).
_base_ = "../s2anet_r50_fpn_1x_rsdd.py"

model = dict(
    bbox_head=dict(
        train_cfg=dict(
            odm_cfg=dict(
                use_dsof=True,
                use_scorer=True,
                use_sala=True,
                consistency_mode='logit_noise',
                repr_mode='sobb',
                scorer_mode='scorer_only',
                lambda_cons=0.0,
                loss_cal_weight=0.0,
                loss_pair_weight=0.0,
                loss_margin_weight=0.0,
            )
        )
    )
)
