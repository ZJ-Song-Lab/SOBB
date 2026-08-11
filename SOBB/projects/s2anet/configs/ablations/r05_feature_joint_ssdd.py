# Series R ablation: Feature-joint scorer (detach candidates only, Series R row 5)
# C2: explicit flags confirm DSOF and SALA are active for prediction-aligned scoring.
_base_ = "../roitransformer_r50_fpn_1x_ssdd.py"
model = dict(
    rbbox_head=dict(
        use_dsof=True,
        use_scorer=True,
        use_sala=True,
        consistency_mode='logit_noise',
        repr_mode='sobb',
        scorer_mode='feature_joint',
        lambda_cons=0.0,
        loss_cal_weight=1.0,
        loss_pair_weight=1.0,
        loss_margin_weight=1.0,
    )
)
