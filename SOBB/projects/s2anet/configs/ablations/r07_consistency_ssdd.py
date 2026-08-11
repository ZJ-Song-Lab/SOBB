# Series R ablation: Consistency-regularized scorer (lambda_cons=0.1, Series R row 7)
#
# C3 documentation: The consistency_mode flag is set to 'logit_noise',
# which applies weak geometric noise (small rotation ~1 deg + translation
# ~0.5 px) to decoded OBB candidates and logit-space noise to the scorer
# output as a proxy for the perturbed forward pass.  A real dual-forward
# (second image-domain forward + geometric inverse transform + permutation
# matching) requires network-level support; see sobb_head.py loss() for the
# proxy implementation and the TODO for the full dual_forward path.
#
# Usage: python tools/run_net.py --config-file projects/s2anet/configs/ablations/r07_consistency_ssdd.py --seed 1
#   Use --seed {1,2,3,4,5} for the five-seed protocol.

_base_ = "../roitransformer_r50_fpn_1x_ssdd.py"

model = dict(
    rbbox_head=dict(
        use_dsof=True,
        use_scorer=True,
        use_sala=True,
        consistency_mode='logit_noise',
        repr_mode='sobb',
        scorer_mode='scorer_only',
        lambda_cons=0.1,
        loss_cal_weight=1.0,
        loss_pair_weight=1.0,
        loss_margin_weight=1.0,
    )
)
