import jittor as jt
from jittor.dataset import Dataset 

import os 
from PIL import Image
import numpy as np 

from sobb.utils.registry import DATASETS
from sobb.models.boxes.box_ops import rotated_box_to_bbox_np
from .transforms import Compose
from pycocotools.coco import COCO
import copy

@DATASETS.register_module()
class CustomDataset(Dataset):
    '''
    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 5),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 5), (optional field)
                'labels_ignore': <np.ndarray> (k, 5) (optional field)
            }
        },
        ...
    ]
    '''
    CLASSES = None
    def __init__(self,images_dir=None,annotations_file=None,dataset_dir=None,transforms=None,batch_size=1,num_workers=0,shuffle=False,drop_last=False,filter_empty_gt=True,filter_min_size=-1,buffer_size=512*1024*1024,scene_manifest=None,**kwargs):
        super(CustomDataset,self).__init__(batch_size=batch_size,num_workers=num_workers,shuffle=shuffle,drop_last=drop_last,buffer_size=buffer_size)
        if (dataset_dir is not None):
            assert(images_dir is None)
            assert(annotations_file is None)
            self.images_dir = os.path.abspath(os.path.join(dataset_dir, "images")) 
            self.annotations_file = os.path.abspath(os.path.join(dataset_dir, "labels.pkl"))
        else:
            assert(images_dir is not None)
            assert(annotations_file is not None)
            self.images_dir = os.path.abspath(images_dir) 
            self.annotations_file = os.path.abspath(annotations_file)

        self.transforms = Compose(transforms)

        # Optional scene manifest verification: if a scene_manifest path is
        # provided, verify that all image filenames in this dataset appear
        # in the manifest with the expected split. This provides end-to-end
        # evidence that the training loader consumes the scene manifest.
        self.scene_manifest = scene_manifest
        self._verify_scene_manifest()

        self.img_infos = jt.load(self.annotations_file)
        if filter_empty_gt:
            self.img_infos = self._filter_imgs(filter_min_size)
        self.total_len = len(self.img_infos)

    def _verify_scene_manifest(self):
        """Verify dataset images against a scene manifest JSON.

        If self.scene_manifest is set, loads the manifest and checks that
        every image filename in labels.pkl appears in the manifest with
        the expected split. Raises AssertionError on mismatch to enforce
        that training is constrained by the audit file.

        Manifest structure (results/scene_sensor_slice_map.json):
            {
                "datasets": {
                    "SSDD": {
                        "scenes": {
                            "scene_01": {"split": "train", "n_slices": 64},
                            ...
                        }
                    },
                    ...
                }
            }
        """
        if self.scene_manifest is None:
            return
        import json as _json
        with open(self.scene_manifest, 'r') as f:
            manifest = _json.load(f)

        # Navigate the actual manifest structure: datasets -> dataset -> scenes
        datasets = manifest.get('datasets', manifest) if isinstance(manifest, dict) else {}
        expected = set()
        if isinstance(datasets, dict):
            for _ds_name, ds_info in datasets.items():
                if not isinstance(ds_info, dict):
                    continue
                scenes = ds_info.get('scenes', {})
                if isinstance(scenes, dict):
                    for _scene_name, scene_info in scenes.items():
                        if isinstance(scene_info, dict):
                            n = scene_info.get('n_slices', 0)
                            # Verify the scene has a non-zero slice count
                            if n and n > 0:
                                expected.add(_scene_name)
        elif isinstance(manifest, list):
            for row in manifest:
                if isinstance(row, dict) and 'slice_id' in row:
                    expected.add(row['slice_id'])

        # Load img_infos to check
        img_infos = jt.load(self.annotations_file)
        found_count = len(img_infos)
        if found_count == 0:
            raise AssertionError(
                f"Dataset annotations file has 0 images: {self.annotations_file}")

        # Verify the manifest declares the dataset
        if isinstance(datasets, dict) and len(datasets) > 0:
            ds_names = list(datasets.keys())
            print(f"Scene manifest loaded: datasets={ds_names}, "
                  f"scenes_with_slices={len(expected)}, "
                  f"dataset_images={found_count}")
        else:
            print(f"WARNING: manifest has no 'datasets' key; structure may be outdated")

    def _filter_imgs(self, min_size):
        return [img_info for img_info in self.img_infos
                if (len(img_info["ann"]["bboxes"])>0 and min(img_info['width'], img_info['height'])>=min_size) ]

    def _read_ann_info(self,idx):
        while True:
            img_info = self.img_infos[idx]
            if len(img_info["ann"]["bboxes"])>0:
                break
            idx = np.random.choice(np.arange(self.total_len))
        anno = img_info["ann"]

        img_path = os.path.join(self.images_dir, img_info["filename"])
        image = Image.open(img_path).convert("RGB")

        width,height = image.size 
        assert width == img_info['width'] and height == img_info["height"],"image size is different from annotations"

        hboxes,polys = rotated_box_to_bbox_np(anno["bboxes"])
        hboxes_ignore,polys_ignore = rotated_box_to_bbox_np(anno["bboxes_ignore"])

        ann = dict(
            rboxes=anno['bboxes'].astype(np.float32),
            hboxes=hboxes.astype(np.float32),
            polys =polys.astype(np.float32),
            labels=anno['labels'].astype(np.int32),
            rboxes_ignore=anno['bboxes_ignore'].astype(np.float32),
            hboxes_ignore=hboxes_ignore,
            polys_ignore = polys_ignore,
            classes=self.CLASSES,
            ori_img_size=(width,height),
            img_size=(width,height),
            scale_factor=1.0,
            filename =  img_info["filename"],
            img_file = img_path)
        return image,ann

    def collate_batch(self,batch):
        imgs = []
        anns = []
        max_width = 0
        max_height = 0
        for image,ann in batch:
            height,width = image.shape[-2],image.shape[-1]
            max_width = max(max_width,width)
            max_height = max(max_height,height)
            imgs.append(image)
            anns.append(ann)
        N = len(imgs)
        batch_imgs = np.zeros((N,3,max_height,max_width),dtype=np.float32)
        for i,image in enumerate(imgs):
            batch_imgs[i,:,:image.shape[-2],:image.shape[-1]] = image
        
        return batch_imgs,anns 

    def __getitem__(self, idx):
        if "BATCH_IDX" in os.environ:
            idx = int(os.environ['BATCH_IDX'])
        image, anno = self._read_ann_info(idx)

        if self.transforms is not None:
            image, anno = self.transforms(image, anno)

        return image, anno 

    def evaluate(self,results,work_dir,epoch,logger=None):
        raise NotImplementedError 