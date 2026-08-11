import cv2
import argparse
import os
import shutil
from sobb.config import init_cfg, get_cfg
from sobb.data.devkits.ImgSplit_multi_process import process
from sobb.data.devkits.convert_data_to_mmdet import convert_data_to_mmdet
from sobb.data.devkits.fair_to_dota import fair_to_dota
from sobb.utils.general import is_win

from sobb.data.devkits.ssdd_to_dota import ssdd_to_dota


def clear(cfg):
    if is_win():
        shutil.rmtree(os.path.join(cfg.source_dataset_path, 'trainval'),ignore_errors=True)
        shutil.rmtree(os.path.join(cfg.target_dataset_path),ignore_errors=True)
    else:
        os.system(f"rm -rf {os.path.join(cfg.source_dataset_path, 'trainval')}")
        os.system(f"rm -rf {os.path.join(cfg.target_dataset_path)}")

def run(cfg):
    if cfg.type=='SSDD+' or cfg.type=='SSDD':
        for task in cfg.convert_tasks:
            print('==============')
            print("convert to dota:", task)
            out_path = os.path.join(cfg.target_dataset_path, task)
            if task == 'test':
                out_path = os.path.join(cfg.target_dataset_path, 'val')
            out_path += '_' + str(cfg.resize)

            # trainval: merge train + val source directories into a
            # temporary combined directory before conversion.
            # This supports the paper's claim that the final model is
            # trained on 928 train+val images (SSDD scenes 1-9).
            if task == 'trainval':
                merged_img = os.path.join(cfg.source_dataset_path, 'JPEGImages_trainval')
                merged_ann = os.path.join(cfg.source_dataset_path, 'Annotations_trainval')
                os.makedirs(merged_img, exist_ok=True)
                os.makedirs(merged_ann, exist_ok=True)
                for sub in ['train', 'val']:
                    src_img = os.path.join(cfg.source_dataset_path, f'JPEGImages_{sub}')
                    src_ann = os.path.join(cfg.source_dataset_path, f'Annotations_{sub}')
                    if not os.path.exists(src_img):
                        print(f'  WARNING: {src_img} does not exist, skipping {sub}')
                        continue
                    for fname in os.listdir(src_img):
                        src_f = os.path.join(src_img, fname)
                        dst_f = os.path.join(merged_img, fname)
                        if not os.path.exists(dst_f):
                            shutil.copy2(src_f, dst_f)
                    if os.path.exists(src_ann):
                        for fname in os.listdir(src_ann):
                            src_f = os.path.join(src_ann, fname)
                            dst_f = os.path.join(merged_ann, fname)
                            if not os.path.exists(dst_f):
                                shutil.copy2(src_f, dst_f)
                img_dir = merged_img
                ann_dir = merged_ann
            else:
                img_dir = os.path.join(cfg.source_dataset_path, f'JPEGImages_{task}')
                ann_dir = os.path.join(cfg.source_dataset_path, f'Annotations_{task}')

            if cfg.type=='SSDD+':
                ssdd_to_dota(
                    img_dir,
                    ann_dir,
                    out_path,
                    cfg.resize,
                    plus=True
                )
            else:
                ssdd_to_dota(
                    img_dir,
                    ann_dir,
                    out_path,
                    cfg.resize,
                    plus=False
                )

            convert_data_to_mmdet(out_path, os.path.join(out_path, 'labels.pkl'), type=cfg.type)

            # Clean up merged trainval temp directories after conversion
            if task == 'trainval':
                shutil.rmtree(merged_img, ignore_errors=True)
                shutil.rmtree(merged_ann, ignore_errors=True)
        return

    if (cfg.type=='FAIR' or cfg.type=='FAIR1M_1_5'):
        for task in cfg.convert_tasks:
            print('==============')
            print("convert to dota:", task)
            fair_to_dota(os.path.join(cfg.source_fair_dataset_path, task), os.path.join(cfg.source_dataset_path, task))

    for task in cfg.tasks:
        label = task.label
        cfg_ = task.config
        print('==============')
        print("processing", label)

        subimage_size=600 if cfg_.subimage_size is None else cfg_.subimage_size
        overlap_size=150 if cfg_.overlap_size is None else cfg_.overlap_size
        multi_scale=[1.] if cfg_.multi_scale is None else cfg_.multi_scale
        horizontal_flip=False if cfg_.horizontal_flip is None else cfg_.horizontal_flip
        vertical_flip=False if cfg_.vertical_flip is None else cfg_.vertical_flip
        rotation_angles=[0.] if cfg_.rotation_angles is None else cfg_.rotation_angles
        assert(rotation_angles == [0.]) #TODO support multi angles
        assert(horizontal_flip == False) #TODO support horizontal_flip
        assert(vertical_flip == False) #TODO support vertical_flip

        assert(label in ['trainval', 'train', 'val', 'test'])
        in_path = os.path.join(cfg.source_dataset_path, label)
        out_path = os.path.join(cfg.target_dataset_path, label)
        # generate trainval
        if (label == 'trainval' and (not os.path.exists(in_path))):
            out_img_path = os.path.join(cfg.source_dataset_path, 'trainval', 'images')
            out_label_path = os.path.join(cfg.source_dataset_path, 'trainval', 'labelTxt')
            os.makedirs(out_img_path,exist_ok=True)
            os.makedirs(out_label_path,exist_ok=True)
            # TODO support Windows etc.
            if is_win():
                shutil.copytree(os.path.join(cfg.source_dataset_path, 'train', 'images'),out_img_path,dirs_exist_ok=True) 
                shutil.copytree(os.path.join(cfg.source_dataset_path, 'val', 'images'),out_img_path,dirs_exist_ok=True)
                shutil.copytree(os.path.join(cfg.source_dataset_path, 'train', 'labelTxt'),out_label_path,dirs_exist_ok=True)
                shutil.copytree(os.path.join(cfg.source_dataset_path, 'val', 'labelTxt'),out_label_path,dirs_exist_ok=True)
            else:
                os.system(f"cp {os.path.join(cfg.source_dataset_path, 'train', 'images', '*')} {out_img_path}")
                os.system(f"cp {os.path.join(cfg.source_dataset_path, 'val', 'images', '*')} {out_img_path}")
                os.system(f"cp {os.path.join(cfg.source_dataset_path, 'train', 'labelTxt', '*')} {out_label_path}")
                os.system(f"cp {os.path.join(cfg.source_dataset_path, 'val', 'labelTxt', '*')} {out_label_path}")
        target_path = process(in_path, out_path, subsize=subimage_size, gap=overlap_size, rates=multi_scale)
        if (label != "test"):
            print("converting to mmdet format...")
            print(cfg.type)
            convert_data_to_mmdet(target_path, os.path.join(target_path, 'labels.pkl'), type=cfg.type)

def main():
    parser = argparse.ArgumentParser(description="Jittor DOTA data preprocess")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--clear",
        default=False,
        action='store_true'
    )
    args = parser.parse_args()
    if args.config_file:
        init_cfg(args.config_file)
    cfg = get_cfg()
    print(cfg.dump())

    if (args.clear):
        clear(cfg)
    else:
        run(cfg)

if __name__ == "__main__":
    main()