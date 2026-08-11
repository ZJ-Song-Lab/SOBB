"""
Reproducible Training Methods for SOBB Paper Experiments
=========================================================

This script provides the exact training and evaluation commands needed to
reproduce every experiment reported in the SOBB manuscript. It contains NO
pre-computed result numbers -- all metrics must be obtained by running the
training commands below on a GPU.

Prerequisites
-------------
1. Jittor + JDet framework installed (see python/sobb/__init__.py)
2. SSDD and RSDD datasets preprocessed via tools/preprocess.py
3. All config files under projects/s2anet/configs/

Five-Seed Protocol
------------------
Every experiment uses seeds {1, 2, 3, 4, 5}. Each seed gets its own
work_dir (work_dirs/{config}_seed{N}) and --no-resume to guarantee
independent training runs. After training, use results/aggregate_seeds.py
to compute mean, std, and paired differences across seeds.

Usage
-----
    python results/reproduce_experiments.py --list
    python results/reproduce_experiments.py --group main
    python results/reproduce_experiments.py --group ablation
    python results/reproduce_experiments.py --group representation
    python results/reproduce_experiments.py --group perturbation
    python results/reproduce_experiments.py --group threshold
    python results/reproduce_experiments.py --group aspect_ratio
    python results/reproduce_experiments.py --group paired
    python results/reproduce_experiments.py --all
"""

import argparse
import os
import subprocess
import sys

SEEDS = [1, 2, 3, 4, 5]
RUN_NET = os.path.join("tools", "run_net.py")
PYTHON = sys.executable


def _work_dir(config_file, seed):
    """Derive per-seed work_dir from config file name."""
    base = os.path.splitext(os.path.basename(config_file))[0]
    return os.path.join("work_dirs", f"{base}_seed{seed}")


def _train_cmd(config_file, seed):
    """Build a training command with per-seed work_dir and --no-resume."""
    wd = _work_dir(config_file, seed)
    return [
        PYTHON, RUN_NET,
        "--config-file", config_file,
        "--seed", str(seed),
        "--work-dir", wd,
        "--no-resume",
    ]


def _eval_cmd(config_file, seed):
    """Build an evaluation command that loads the seed-specific checkpoint."""
    wd = _work_dir(config_file, seed)
    return [
        PYTHON, RUN_NET,
        "--config-file", config_file,
        "--task", "test",
        "--work-dir", wd,
        "--checkpoint", "latest",
    ]


def _eval_cmd_cfg(config_file, seed, cfg_options=None):
    """Build an evaluation command with optional config overrides."""
    cmd = _eval_cmd(config_file, seed)
    if cfg_options:
        cmd.append("--cfg-options")
        cmd.extend(cfg_options)  # cfg_options is list of key=value strings without --cfg-options
    return cmd


# ------------------------------------------------------------------
# Group 1: Main Results (Table 6 & Table 7)
#   - SSDD/RSDD x two-stage/single-stage
#   - Five seeds per cell with independent work_dirs
# ------------------------------------------------------------------
MAIN_CONFIGS = {
    "SSDD_two_stage": "projects/s2anet/configs/roitransformer_r50_fpn_1x_ssdd.py",
    "RSDD_two_stage": "projects/s2anet/configs/roitransformer_r50_fpn_1x_rsdd.py",
    "SSDD_single_stage": "projects/s2anet/configs/s2anet_r50_fpn_1x_ssdd.py",
    "RSDD_single_stage": "projects/s2anet/configs/s2anet_r50_fpn_1x_rsdd.py",
}

FINAL_CONFIGS = {
    "SSDD_two_stage_trainval": "projects/s2anet/configs/roitransformer_r50_fpn_1x_ssdd_trainval.py",
    "RSDD_two_stage_trainval": "projects/s2anet/configs/roitransformer_r50_fpn_1x_rsdd_trainval.py",
}


def commands_main():
    """Table 6 (SSDD) and Table 7 (RSDD): Ours-Two and Ours-Single.

    Each cell trains 5 independent seeds, then evaluates each seed
    separately. Use results/aggregate_seeds.py to compute mean/std.
    """
    cmds = []
    for label, cfg in MAIN_CONFIGS.items():
        for seed in SEEDS:
            cmds.append((_train_cmd(cfg, seed),
                         "Train {} seed={} (independent work_dir)".format(label, seed)))
        for seed in SEEDS:
            cmds.append((_eval_cmd(cfg, seed),
                         "Eval {} seed={}".format(label, seed)))
    return cmds


def commands_final():
    """Final models trained on train+val splits (paper Section 5.1).

    SSDD: 928 images (train+val), RSDD: full train+val.
    """
    cmds = []
    for label, cfg in FINAL_CONFIGS.items():
        for seed in SEEDS:
            cmds.append((_train_cmd(cfg, seed),
                         "Train {} (trainval) seed={}".format(label, seed)))
        for seed in SEEDS:
            cmds.append((_eval_cmd(cfg, seed),
                         "Eval {} (trainval) seed={}".format(label, seed)))
    return cmds


# ------------------------------------------------------------------
# Group 2: Ablation Study (Table 1, R01-R07)
# ------------------------------------------------------------------
ABLATION_CONFIGS = {
    "R01": ("r01_gt_bce_ssdd.py", "GT-candidate independent BCE"),
    "R02": ("r02_pred_aligned_ssdd.py", "Prediction-aligned calibration"),
    "R03": ("r03_pairwise_ssdd.py", "Pairwise scoring"),
    "R04": ("r04_risk_aware_ssdd.py", "Risk-aware, scorer-only (CORE)"),
    "R05": ("r05_feature_joint_ssdd.py", "Risk-aware, feature-joint"),
    "R06": ("r06_fully_joint_ssdd.py", "Risk-aware, fully joint"),
    "R07": ("r07_consistency_ssdd.py", "Risk-aware + consistency"),
}


def commands_ablation():
    """Table 1 ablation rows R01-R07 on SSDD, five seeds each."""
    base = os.path.join("projects", "s2anet", "configs", "ablations")
    cmds = []
    for paper_id, (fname, desc) in ABLATION_CONFIGS.items():
        cfg = os.path.join(base, fname)
        for seed in SEEDS:
            cmds.append((_train_cmd(cfg, seed),
                         "Train {} ({}) seed={}".format(paper_id, desc, seed)))
        for seed in SEEDS:
            cmds.append((_eval_cmd(cfg, seed),
                         "Eval {} ({}) seed={}".format(paper_id, desc, seed)))
    return cmds


# ------------------------------------------------------------------
# Group 3: Representation Transfer (Table T)
#   Angle / GWD / KFIoU / COBB / GT-candidate-BCE-SOBB / Risk-aware-SOBB
#   on RSDD single-stage
#
# S0-8: Each representation uses a dedicated config file that properly
# swaps the regression head, bbox coder, and loss function, rather than
# attempting runtime --cfg-options overrides that the head constructor
# does not accept. The base single-stage config is s2anet_r50_fpn_1x_rsdd.py;
# representation variants inherit from it and override the relevant model
# fields. COBB is a four-candidate convex hull representation (not a string).
# ------------------------------------------------------------------
REPR_CONFIGS = {
    "Angle": "projects/s2anet/configs/repr/s2anet_r50_fpn_1x_rsdd_angle.py",
    "GWD": "projects/s2anet/configs/repr/s2anet_r50_fpn_1x_rsdd_gwd.py",
    "KFIoU": "projects/s2anet/configs/repr/s2anet_r50_fpn_1x_rsdd_kfiou.py",
    "COBB": "projects/s2anet/configs/repr/s2anet_r50_fpn_1x_rsdd_cobb.py",
    "COBB_direct_area": "projects/s2anet/configs/repr/s2anet_r50_fpn_1x_rsdd_cobb_direct_area.py",
    "GT-candidate_BCE-SOBB": "projects/s2anet/configs/repr/s2anet_r50_fpn_1x_rsdd_gt_bce.py",
    "Risk-aware_SOBB": "projects/s2anet/configs/s2anet_r50_fpn_1x_rsdd.py",
}


def commands_representation():
    """Table T: representation comparison on RSDD.

    Each representation uses a dedicated config file that properly
    swaps the regression head, bbox coder, and loss function, rather
    than attempting runtime --cfg-options overrides that the head
    constructor does not accept.
    """
    cmds = []
    for rep_name, cfg in REPR_CONFIGS.items():
        for seed in SEEDS:
            rep_slug = rep_name.replace('-', '_').replace(' ', '_')
            wd = os.path.join("work_dirs",
                f"repr_{rep_slug}_seed{seed}")
            cmd = [
                PYTHON, RUN_NET,
                "--config-file", cfg,
                "--seed", str(seed),
                "--work-dir", wd,
                "--no-resume",
            ]
            cmds.append((cmd,
                         "Train {} seed={}".format(rep_name, seed)))
        for seed in SEEDS:
            rep_slug = rep_name.replace('-', '_').replace(' ', '_')
            wd = os.path.join("work_dirs",
                f"repr_{rep_slug}_seed{seed}")
            cmd = [
                PYTHON, RUN_NET,
                "--config-file", cfg,
                "--seed", str(seed),
                "--task", "test",
                "--work-dir", wd,
                "--checkpoint", "latest",
            ]
            cmds.append((cmd,
                         "Eval {} seed={}".format(rep_name, seed)))
    return cmds


# ------------------------------------------------------------------
# Group 4: Perturbation Diagnostics
#   angle-jitter {1, 3} deg, translation {1, 2} px, Gamma {0.8, 1.0, 1.2}
#   scale-jitter {0.8x, 1.0x, 1.2x}
#   on SSDD and RSDD two-stage
#   Uses --cfg-options to set test-time perturbation parameters
# ------------------------------------------------------------------
def commands_perturbation():
    """Robustness stress tests: angle-jitter and scale-jitter sweeps."""
    datasets = {
        "SSDD": "projects/s2anet/configs/roitransformer_r50_fpn_1x_ssdd.py",
        "RSDD": "projects/s2anet/configs/roitransformer_r50_fpn_1x_rsdd.py",
    }
    angle_mags = [1, 3]
    trans_mags = [1, 2]
    gamma_mags = [0.8, 1.0, 1.2]
    scale_mags = [0.8, 1.0, 1.2]
    cmds = []
    for ds, cfg in datasets.items():
        for ang in angle_mags:
            for seed in SEEDS:
                opts = ["model.test_cfg.rcnn.test_angle_jitter={}".format(ang)]
                cmds.append((_eval_cmd_cfg(cfg, seed, opts),
                             "Eval {} angle-jitter={}deg seed={}".format(ds, ang, seed)))
        for sc in scale_mags:
            for seed in SEEDS:
                opts = ["model.test_cfg.rcnn.test_scale_jitter={}".format(sc)]
                cmds.append((_eval_cmd_cfg(cfg, seed, opts),
                             "Eval {} scale-jitter={}x seed={}".format(ds, sc, seed)))
        for tr in trans_mags:
            for seed in SEEDS:
                opts = ["model.test_cfg.rcnn.test_trans_jitter={}".format(tr)]
                cmds.append((_eval_cmd_cfg(cfg, seed, opts),
                             "Eval {} trans-jitter={}px seed={}".format(ds, tr, seed)))
        for ga in gamma_mags:
            for seed in SEEDS:
                opts = ["model.test_cfg.rcnn.test_gamma_jitter={}".format(ga)]
                cmds.append((_eval_cmd_cfg(cfg, seed, opts),
                             "Eval {} gamma-jitter={} seed={}".format(ds, ga, seed)))
    return cmds


# ------------------------------------------------------------------
# Group 5: NMS Threshold Sweep
#   IoU NMS thresholds: {0.05, 0.10, 0.20, 0.30, 0.50}
#   on SSDD and RSDD two-stage
# ------------------------------------------------------------------
def commands_threshold():
    """IoU NMS threshold sweep to find optimal threshold."""
    datasets = {
        "SSDD": "projects/s2anet/configs/roitransformer_r50_fpn_1x_ssdd.py",
        "RSDD": "projects/s2anet/configs/roitransformer_r50_fpn_1x_rsdd.py",
    }
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
    cmds = []
    for ds, cfg in datasets.items():
        for thr in thresholds:
            for seed in SEEDS:
                opts = ["model.test_cfg.rcnn.nms.iou_thr={}".format(thr)]
                cmds.append((_eval_cmd_cfg(cfg, seed, opts),
                             "Eval {} nms_iou_thr={} seed={}".format(ds, thr, seed)))
    return cmds


# ------------------------------------------------------------------
# Group 6: Aspect Ratio Diagnostics
#   Per-bin metrics for aspect ratio bins
# ------------------------------------------------------------------
def commands_aspect_ratio():
    """Per-aspect-ratio-bin evaluation diagnostics."""
    datasets = {
        "SSDD": "projects/s2anet/configs/roitransformer_r50_fpn_1x_ssdd.py",
        "RSDD": "projects/s2anet/configs/roitransformer_r50_fpn_1x_rsdd.py",
    }
    cmds = []
    for ds, cfg in datasets.items():
        for seed in SEEDS:
            opts = ["model.test_cfg.rcnn.eval_by_aspect_ratio=True"]
            cmds.append((_eval_cmd_cfg(cfg, seed, opts),
                         "Eval {} per-aspect-ratio-bin seed={}".format(ds, seed)))
    return cmds


# ------------------------------------------------------------------
# Group 7: Paired Differences
#   Risk-aware vs prediction-aligned calibration
#   Metrics: mAP75, mAP50_95, A_sel, Brier, ECE, R_sel, R_tie
#   Uses --cfg-options to enable calibration computation
# ------------------------------------------------------------------
def commands_paired():
    """Paired comparison: R04 (risk-aware) vs R02 (prediction-aligned)."""
    base = os.path.join("projects", "s2anet", "configs", "ablations")
    r02_cfg = os.path.join(base, "r02_pred_aligned_ssdd.py")
    r04_cfg = os.path.join(base, "r04_risk_aware_ssdd.py")
    cmds = []
    for seed in SEEDS:
        cmds.append((_train_cmd(r02_cfg, seed),
                     "Train R02 (pred-aligned) seed={}".format(seed)))
        cmds.append((_train_cmd(r04_cfg, seed),
                     "Train R04 (risk-aware) seed={}".format(seed)))
    for seed in SEEDS:
        opts = ["model.test_cfg.rcnn.compute_calibration=True"]
        cmds.append((_eval_cmd_cfg(r02_cfg, seed, opts),
                     "Eval R02 calibration metrics seed={}".format(seed)))
        cmds.append((_eval_cmd_cfg(r04_cfg, seed, opts),
                     "Eval R04 calibration metrics seed={}".format(seed)))
    return cmds


def commands_calibration():
    """Calibration metrics: ECE, Brier, regret, tie/switch rates."""
    base = os.path.join("projects", "s2anet", "configs", "ablations")
    r02_cfg = os.path.join(base, "r02_pred_aligned_ssdd.py")
    r04_cfg = os.path.join(base, "r04_risk_aware_ssdd.py")
    r07_cfg = os.path.join(base, "r07_consistency_ssdd.py")
    cmds = []
    for cfg, label in [(r02_cfg, "R02"), (r04_cfg, "R04"), (r07_cfg, "R07")]:
        for seed in SEEDS:
            opts = ["model.test_cfg.rcnn.compute_calibration=True"]
            cmds.append((_eval_cmd_cfg(cfg, seed, opts),
                         "Eval {} calibration (ECE/Brier/regret) seed={}".format(label, seed)))
    return cmds


GROUPS = {
    "main": ("Table 6 & 7: Main results (SSDD/RSDD x two/single-stage)", commands_main),
    "final": ("Final train+val models (paper Section 5.1)", commands_final),
    "ablation": ("Table 1: Ablation study R01-R07", commands_ablation),
    "representation": ("Table T: Representation transfer comparison", commands_representation),
    "perturbation": ("Perturbation diagnostics: angle/scale jitter", commands_perturbation),
    "threshold": ("NMS IoU threshold sweep", commands_threshold),
    "aspect_ratio": ("Per-aspect-ratio-bin diagnostics", commands_aspect_ratio),
    "paired": ("Paired differences: risk-aware vs pred-aligned", commands_paired),
    "calibration": ("Calibration diagnostics: ECE, Brier, regret, tie/switch", commands_calibration),
}


def print_commands(cmds, dry_run=True):
    for cmd, desc in cmds:
        print("\n# {}".format(desc))
        print(" ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", choices=list(GROUPS.keys()),
                        help="Which experiment group to reproduce")
    parser.add_argument("--all", action="store_true",
                        help="Print commands for all groups")
    parser.add_argument("--list", action="store_true",
                        help="List all experiment groups")
    parser.add_argument("--run", action="store_true",
                        help="Actually execute commands (default: dry-run)")
    args = parser.parse_args()

    if args.list:
        print("Available experiment groups:")
        for key, (desc, _) in GROUPS.items():
            print("  {:20s}  {}".format(key, desc))
        return

    if args.all:
        for key, (desc, fn) in GROUPS.items():
            print("\n{}".format("=" * 60))
            print("GROUP: {} -- {}".format(key, desc))
            print("{}".format("=" * 60))
            print_commands(fn(), dry_run=not args.run)
        return

    if args.group:
        desc, fn = GROUPS[args.group]
        print("\n{}".format("=" * 60))
        print("GROUP: {} -- {}".format(args.group, desc))
        print("{}".format("=" * 60))
        print_commands(fn(), dry_run=not args.run)
        return

    parser.print_help()


def print_aggregation_instructions():
    """Print instructions for aggregating per-seed results."""
    print("\n" + "=" * 60)
    print("AGGREGATION")
    print("=" * 60)
    print("After training and evaluating all 5 seeds, aggregate results:")
    print("  python results/aggregate_seeds.py \\\n"
          "    --config work_dirs/roitransformer_r50_fpn_1x_ssdd \\\n"
          "    --output work_dirs/roitransformer_r50_fpn_1x_ssdd/summary.json")
    print("For paired comparisons:")
    print("  python results/aggregate_seeds.py \\\
"
          "    --config work_dirs/roitransformer_r50_fpn_1x_ssdd \\\
"
          "    --paired work_dirs/r04_risk_aware_ssdd \\\
"
          "    --metric eval/0_meanAP")


if __name__ == "__main__":
    main()
    print_aggregation_instructions()
