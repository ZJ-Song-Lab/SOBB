"""
Seed Aggregation Script for Five-Seed Protocol
================================================

Reads per-seed evaluation JSON outputs from work_dirs/{config}_seed{N}/
and computes mean, sample standard deviation, and paired differences
across the five seeds. Produces a summary table for paper tables.

The evaluation JSON is produced by Runner.test() which now calls
dataset.evaluate() and saves results to test/eval_{epoch}.json.

Usage:
    python results/aggregate_seeds.py --config work_dirs/roitransformer_r50_fpn_1x_ssdd
    python results/aggregate_seeds.py --config work_dirs/roitransformer_r50_fpn_1x_ssdd --paired work_dirs/r02_pred_aligned_ssdd
"""

import argparse
import os
import json
import glob
import numpy as np


SEEDS = [1, 2, 3, 4, 5]


def load_seed_results(work_dir_pattern):
    """Load evaluation results from per-seed work directories.

    Reads test/eval_*.json files produced by Runner.test() -> evaluate().
    Falls back to textlog/*.txt parsing if JSON is unavailable.

    Args:
        work_dir_pattern: e.g. "work_dirs/roitransformer_r50_fpn_1x_ssdd"
                          (will append _seed{N} for each seed)

    Returns:
        dict: seed -> {metric -> value}
    """
    results = {}
    for seed in SEEDS:
        wd = f"{work_dir_pattern}_seed{seed}"
        # Primary: read evaluation JSON (produced by test() -> evaluate())
        eval_jsons = sorted(glob.glob(os.path.join(wd, "test", "eval_*.json")))
        if eval_jsons:
            with open(eval_jsons[-1], 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        results[seed] = {k: v for k, v in data.items()
                                         if k.startswith('eval/')}
                        if results[seed]:
                            continue
                except (json.JSONDecodeError, ValueError):
                    pass

        # Fallback: parse textlog/*.txt for metric lines
        txt_logs = sorted(glob.glob(os.path.join(wd, "textlog", "*.txt")))
        txt_logs += sorted(glob.glob(os.path.join(wd, "*.txt")))
        if txt_logs:
            parsed = _parse_textlog(txt_logs[-1])
            if parsed:
                results[seed] = parsed
                continue

        print(f"  WARN: no eval results found for seed {seed} in {wd}")

    return results


def _parse_textlog(log_path):
    """Parse evaluation metrics from a text log file.

    Looks for lines containing 'eval/' metric keys and extracts
    the numeric value following the key.
    """
    metric_keys = [
        'eval/0_meanAP', 'eval/0_meanAP5095', 'eval/0_meanAP75',
        'eval/0_meanAR100', 'eval/0_meanAP_s'
    ]
    result = {}
    try:
        with open(log_path, 'r', errors='ignore') as f:
            for line in f:
                for key in metric_keys:
                    if key in line:
                        parts = line.split(key)
                        if len(parts) > 1:
                            rest = parts[-1].strip()
                            tokens = rest.split()
                            if tokens:
                                try:
                                    val = float(tokens[0].rstrip(',').rstrip(':'))
                                    result[key] = val
                                except ValueError:
                                    pass
    except IOError:
        pass
    return result


def aggregate(results):
    """Compute mean and sample std across seeds.

    Args:
        results: dict seed -> {metric -> value}

    Returns:
        dict: metric -> {mean, std, values}
    """
    all_metrics = set()
    for seed_data in results.values():
        all_metrics.update(seed_data.keys())

    summary = {}
    for metric in sorted(all_metrics):
        values = [results[s][metric] for s in results if metric in results[s]]
        if len(values) < 2:
            summary[metric] = {'mean': float(np.mean(values)) if values else 0.0,
                               'std': 0.0, 'values': values}
        else:
            summary[metric] = {'mean': float(np.mean(values)),
                               'std': float(np.std(values, ddof=1)),
                               'values': values}
    return summary


def paired_difference(results_a, results_b, metric='eval/0_meanAP'):
    """Compute paired difference between two configs across seeds.

    Returns:
        dict with per-seed differences, mean, std, and t-statistic
    """
    diffs = []
    for seed in SEEDS:
        if seed in results_a and seed in results_b:
            if metric in results_a[seed] and metric in results_b[seed]:
                d = results_a[seed][metric] - results_b[seed][metric]
                diffs.append(d)

    if len(diffs) < 2:
        return {'diffs': diffs, 'mean': 0.0, 'std': 0.0, 'n': len(diffs)}

    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    n = len(diffs)
    t_stat = mean_d / (std_d / np.sqrt(n)) if std_d > 0 else 0.0

    return {'diffs': diffs, 'mean': mean_d, 'std': std_d,
            'n': n, 't_stat': t_stat}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                        help="base work_dir pattern (e.g. work_dirs/roitransformer_r50_fpn_1x_ssdd)")
    parser.add_argument("--paired", default=None,
                        help="second config base work_dir for paired comparison")
    parser.add_argument("--metric", default='eval/0_meanAP',
                        help="metric for paired comparison")
    parser.add_argument("--output", default=None,
                        help="output JSON file for summary")
    args = parser.parse_args()

    print(f"\nLoading results for: {args.config}")
    results = load_seed_results(args.config)
    print(f"  Found results for {len(results)} seeds")

    if len(results) == 0:
        print("  No results found. Run training and evaluation first:")
        print(f"  python tools/run_net.py --config-file <cfg> --seed {{1..5}} "
              f"--work-dir {args.config}_seed{{N}} --no-resume")
        print(f"  python tools/run_net.py --config-file <cfg> --task test "
              f"--work-dir {args.config}_seed{{N}} --checkpoint latest")
        print("  Then re-run this script.")
        return

    if len(results) < len(SEEDS):
        missing = [s for s in SEEDS if s not in results]
        print(f"  WARNING: missing results for seeds {missing}")
        print("  Aggregation requires all 5 seeds for valid statistics.")

    summary = aggregate(results)
    print(f"\n{'Metric':<40} {'Mean':>10} {'Std':>10} {'Seeds':>10}")
    print("-" * 75)
    for metric in sorted(summary.keys()):
        s = summary[metric]
        print(f"{metric:<40} {s['mean']:>10.4f} {s['std']:>10.4f} {len(s['values']):>10}")

    if args.paired:
        print(f"\nPaired comparison: {args.config} vs {args.paired}")
        results_b = load_seed_results(args.paired)
        pd = paired_difference(results, results_b, args.metric)
        print(f"  Metric: {args.metric}")
        print(f"  Per-seed diffs: {[f'{d:.4f}' for d in pd['diffs']]}")
        print(f"  Mean diff: {pd['mean']:.4f} +/- {pd['std']:.4f} (n={pd['n']})")
        print(f"  t-statistic: {pd.get('t_stat', 0.0):.3f}")

    if args.output:
        out = {'config': args.config, 'summary': summary}
        if args.paired:
            out['paired'] = {'config': args.paired, 'metric': args.metric,
                             'result': paired_difference(
                                 results, load_seed_results(args.paired), args.metric)}
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
