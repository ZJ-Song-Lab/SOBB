import argparse
import os
import jittor as jt
jt.flags.use_cuda_managed_allocator=1
from sobb.runner import Runner
from sobb.config import init_cfg, get_cfg
from sobb.utils.general import set_random_seed


def _apply_cfg_options(cfg, options):
    """Apply --cfg-options key=value pairs to config dict.

    Example:
        --cfg-options model.rbbox_head.loss_cal_weight=0.0 test_cfg.rcnn.nms.iou_thr=0.15
    """
    if not options:
        return
    for opt in options:
        if '=' not in opt:
            continue
        key, val = opt.split('=', 1)
        keys = key.split('.')
        d = cfg
        for k in keys[:-1]:
            if k not in d or d[k] is None:
                d[k] = {}
            if not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        try:
            v = eval(val)
        except Exception:
            v = val
        d[keys[-1]] = v


def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="train",
        help="train,val,test,vis_test",
        type=str,
    )
    parser.add_argument(
        "--no_cuda",
        action='store_true'
    )
    parser.add_argument(
        "--save_dir",
        default=".",
        type=str,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for five-seed protocol {1,2,3,4,5}",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        type=str,
        help="override config work_dir (required for per-seed independent training directories)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="checkpoint path for evaluation or resume (use 'latest' to auto-find newest)",
    )
    parser.add_argument(
        "--no-resume",
        action='store_true',
        help="disable auto-resume from latest checkpoint (required for independent five-seed training)",
    )
    parser.add_argument(
        "--cfg-options",
        nargs='*',
        default=None,
        help="override config options, e.g. --cfg-options model.rbbox_head.loss_cal_weight=0.0",
    )
    args = parser.parse_args()

    if not args.no_cuda:
        jt.flags.use_cuda = 1

    assert args.task in ["train", "val", "test", "vis_test"], \
        f"{args.task} not support, please choose [train,val,test,vis_test]"

    if args.seed is not None:
        set_random_seed(args.seed)
        print(f"Set random seed to {args.seed}")

    if args.config_file:
        init_cfg(args.config_file)

    cfg = get_cfg()

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif args.seed is not None:
        base = cfg.work_dir.rstrip('/')
        cfg.work_dir = base + f"_seed{args.seed}"

    if args.no_resume:
        cfg.resume_path = "__no_resume__"
    elif args.checkpoint is not None:
        if args.checkpoint == "latest":
            from sobb.utils.general import search_ckpt
            found = search_ckpt(cfg.work_dir)
            if found:
                cfg.resume_path = found
            else:
                print(f"WARNING: no checkpoint found in {cfg.work_dir}")
        else:
            cfg.resume_path = args.checkpoint

    _apply_cfg_options(cfg, args.cfg_options)

    runner = Runner()

    if args.task == "train":
        runner.run()
    elif args.task == "val":
        runner.val()
    elif args.task == "test":
        runner.test()
    elif args.task == "vis_test":
        runner.run_on_images(args.save_dir)

if __name__ == "__main__":
    main()
