from numpy.lib import save
from sobb.data.devkits.voc_eval import voc_eval_dota
from sobb.models.boxes.box_ops import rotated_box_to_poly_np, rotated_box_to_poly_single
from sobb.utils.general import check_dir
from sobb.utils.registry import DATASETS
from sobb.config.constant import get_classes_by_name
from sobb.data.custom import CustomDataset
from sobb.ops.nms_poly import iou_poly
import os
import jittor as jt
import numpy as np
from tqdm import tqdm

def s2anet_post(result):
    dets,labels = result 
    labels = labels+1 
    scores = dets[:,5]
    dets = dets[:,:5]
    polys = rotated_box_to_poly_np(dets)
    return polys,scores,labels

@DATASETS.register_module()
class DOTADataset(CustomDataset):

    def __init__(self,*arg,balance_category=False,version='1',**kwargs):
        assert version in ['1', '1_5', '2']
        self.CLASSES = get_classes_by_name('DOTA'+version)
        super().__init__(*arg,**kwargs)
        if balance_category:
            self.img_infos = self._balance_categories()
            self.total_len = len(self.img_infos)

    def _balance_categories(self):
        img_infos = self.img_infos
        cate_dict = {}
        for idx,img_info in enumerate(img_infos):
            unique_labels = np.unique(img_info["ann"]["labels"])
            for label in unique_labels:
                if label not in cate_dict:
                    cate_dict[label]=[]
                cate_dict[label].append(idx)
        new_idx = []
        balance_dict={
            "storage-tank":(1,526),
            "baseball-diamond":(2,202),
            "ground-track-field":(1,575),
            "swimming-pool":(2,104),
            "soccer-ball-field":(1,962),
            "roundabout":(1,711),
            "tennis-court":(1,655),
            "basketball-court":(4,0),
            "helicopter":(8,0),
            "container-crane":(50,0)
        }

        for k,d in cate_dict.items():
            classname = self.CLASSES[k-1]
            l1,l2 = balance_dict.get(classname,(1,0))
            new_d = d*l1+d[:l2]
            new_idx.extend(new_d)
        img_infos = [self.img_infos[idx] for idx in new_idx]
        return img_infos
    
    def parse_result(self,results,save_path):
        check_dir(save_path)
        data = {}
        for (dets,labels),img_name in results:
            img_name = os.path.splitext(img_name)[0]
            for det,label in zip(dets,labels):
                bbox = det[:5]
                score = det[5]
                classname = self.CLASSES[label]
                bbox = rotated_box_to_poly_single(bbox)
                temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                            img_name, score, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4],
                            bbox[5], bbox[6], bbox[7])
                if classname not in data:
                    data[classname] = []
                data[classname].append(temp_txt)
        for classname,lines in data.items():
            f_out = open(os.path.join(save_path, classname + '.txt'), 'w')
            f_out.writelines(lines)
            f_out.close()

    def evaluate(self,results,work_dir,epoch,logger=None,save=True):
        print("Calculating mAP......")
        if save:
            save_path = os.path.join(work_dir,f"detections/val_{epoch}")
            check_dir(save_path)
            jt.save(results,save_path+"/val.pkl")
        dets = []
        gts = []
        diffcult_polys = {}
        for img_idx,(result,target) in enumerate(results):
            det_polys,det_scores,det_labels =  result
            det_labels = det_labels + 1
            if det_polys.size>0:
                idx1 = np.ones((det_labels.shape[0],1))*img_idx
                det = np.concatenate([idx1,det_polys,det_scores.reshape(-1,1),det_labels.reshape(-1,1)],axis=1)
                dets.append(det)
            
            scale_factor = target["scale_factor"]
            gt_polys = target["polys"].copy()
            # scale_factor is [w_ratio, h_ratio]; apply per-axis to 8-col polys
            if hasattr(scale_factor, '__len__') and len(scale_factor) == 2:
                sf_w, sf_h = float(scale_factor[0]), float(scale_factor[1])
                gt_polys[:, 0::2] = gt_polys[:, 0::2] / sf_w
                gt_polys[:, 1::2] = gt_polys[:, 1::2] / sf_h
            else:
                gt_polys = gt_polys / float(scale_factor)

            if gt_polys.size>0:
                gt_labels = target["labels"].reshape(-1,1)
                idx2 = np.ones((gt_labels.shape[0],1))*img_idx
                gt = np.concatenate([idx2,gt_polys,gt_labels],axis=1)
                gts.append(gt)
            diffcult_polys[img_idx] = target["polys_ignore"]/scale_factor
        if len(dets) == 0:
            aps = {}
            for i,classname in tqdm(enumerate(self.CLASSES),total=len(self.CLASSES)):
                aps["eval/"+str(i+1)+"_"+classname+"_AP"]=0
                aps["eval/"+str(i+1)+"_"+classname+"_mAP5095"]=0
                aps["eval/"+str(i+1)+"_"+classname+"_mAP75"]=0
                aps["eval/"+str(i+1)+"_"+classname+"_AR100"]=0
                aps["eval/"+str(i+1)+"_"+classname+"_mAP_s"]=0
            aps["eval/0_meanAP"]=0
            aps["eval/0_meanAP5095"]=0
            aps["eval/0_meanAP75"]=0
            aps["eval/0_meanAR100"]=0
            aps["eval/0_meanAP_s"]=0
            return aps
        dets = np.concatenate(dets)
        gts = np.concatenate(gts)
        aps = {}
        for i,classname in tqdm(enumerate(self.CLASSES),total=len(self.CLASSES)):
            c_dets = dets[dets[:,-1]==(i+1)][:,:-1]
            c_gts = gts[gts[:,-1]==(i+1)][:,:-1]
            img_idx = gts[:,0].copy()
            classname_gts = {}
            for idx in np.unique(img_idx):
                g = c_gts[c_gts[:,0]==idx,:][:,1:]
                dg = diffcult_polys[idx].copy().reshape(-1,8)
                diffculty = np.zeros(g.shape[0]+dg.shape[0])
                diffculty[int(g.shape[0]):]=1
                diffculty = diffculty.astype(bool)
                g = np.concatenate([g,dg])
                classname_gts[idx] = {"box":g.copy(),"det":[False for i in range(len(g))],'difficult':diffculty.copy()}
            # Multi-IoU evaluation: mAP50:95, mAP75, AR@(1,10,100).
            # IMPORTANT: voc_eval_dota_multi resets GT det state internally,
            # but we must also reset before the standalone AP50 call below.
            from sobb.data.devkits.voc_eval import (
                voc_eval_dota_multi, _reset_gt_det, scale_stratified_eval)
            multi = voc_eval_dota_multi(c_dets, classname_gts)
            # Reset GT state after multi-threshold eval to avoid pollution
            _reset_gt_det(classname_gts)
            rec, prec, ap = voc_eval_dota(c_dets,classname_gts,iou_func=iou_poly)
            aps["eval/"+str(i+1)+"_"+classname+"_AP"]=ap
            aps["eval/"+str(i+1)+"_"+classname+"_mAP5095"]=multi['mAP5095']
            aps["eval/"+str(i+1)+"_"+classname+"_mAP75"]=multi['mAP75']
            aps["eval/"+str(i+1)+"_"+classname+"_AR100"]=multi['AR100']
            # S0-9: Scale-stratified evaluation at .50:.95 (10 thresholds)
            # averaged, matching the paper's mAP_s definition.
            _reset_gt_det(classname_gts)
            _iou_thrs = np.round(np.arange(0.5, 1.0, 0.05), 2)
            _scale_aps = []
            for _ov in _iou_thrs:
                _reset_gt_det(classname_gts)
                _sr = scale_stratified_eval(
                    c_dets, classname_gts, {}, ovthresh=float(_ov))
                _scale_aps.append(_sr.get('small', 0.0))
            _reset_gt_det(classname_gts)
            aps["eval/"+str(i+1)+"_"+classname+"_mAP_s"]=float(np.mean(_scale_aps)) if _scale_aps else 0.0
        # meanAP: average of class-level AP50 only (standard DOTA evaluation)
        ap50_vals = [v for k, v in aps.items() if k.endswith('_AP')]
        mean_ap = sum(ap50_vals) / max(len(ap50_vals), 1)
        # mAP50:95: average of per-class mAP50:95 values
        map5095_vals = [v for k, v in aps.items() if k.endswith('_mAP5095')]
        mean_map5095 = sum(map5095_vals) / max(len(map5095_vals), 1) if map5095_vals else 0.0
        # mAP75: average of per-class mAP75 values
        map75_vals = [v for k, v in aps.items() if k.endswith('_mAP75')]
        mean_map75 = sum(map75_vals) / max(len(map75_vals), 1) if map75_vals else 0.0
        # meanAR: average of per-class AR100 values
        ar_vals = [v for k, v in aps.items() if k.endswith('_AR100')]
        mean_ar = sum(ar_vals) / max(len(ar_vals), 1) if ar_vals else 0.0
        aps["eval/0_meanAP"] = mean_ap
        aps["eval/0_meanAP5095"] = mean_map5095
        aps["eval/0_meanAP75"] = mean_map75
        aps["eval/0_meanAR100"] = mean_ar
        # mAP_s: average of per-class small-object AP
        map_s_vals = [v for k, v in aps.items() if k.endswith('_mAP_s')]
        mean_map_s = sum(map_s_vals) / max(len(map_s_vals), 1) if map_s_vals else 0.0
        aps["eval/0_meanAP_s"] = mean_map_s
        return aps
            
            
@DATASETS.register_module()
class SSDDDataset(DOTADataset):
    """SSDD/SSDD+ dataset for SAR ship detection.

    The data is preprocessed into DOTA format via
    ``sobb.data.devkits.ssdd_to_dota`` (see ``docs/ssdd.md``), so the loading
    logic is identical to ``DOTADataset``. Only the category list differs:
    SSDD/SSDD+ contains a single class ``ship``.
    """
    def __init__(self, *arg, balance_category=False, **kwargs):
        super().__init__(*arg, balance_category=balance_category, version='1', **kwargs)
        self.CLASSES = get_classes_by_name('SSDD')


@DATASETS.register_module()
class RSDDDataset(DOTADataset):
    """RSDD dataset for SAR ship detection.

    RSDD is preprocessed into the same DOTA-style format as SSDD, so the
    loading logic is identical to ``DOTADataset``.  RSDD contains a single
    class ``ship``.
    """
    def __init__(self, *arg, balance_category=False, **kwargs):
        super().__init__(*arg, balance_category=balance_category, version='1', **kwargs)
        self.CLASSES = get_classes_by_name('RSDD')


def test_eval():
    results= jt.load("projects/s2anet/work_dirs/s2anet_r50_fpn_1x_dota/detections/val_0/val.pkl")
    results = jt.load("projects/s2anet/work_dirs/s2anet_r50_fpn_1x_dota/detections/val_rotate_balance/val.pkl")
    # results = results
    dataset = DOTADataset(annotations_file='/mnt/disk/lxl/dataset/DOTA_1024/trainval_split/trainval1024.pkl',
        images_dir='/mnt/disk/lxl/dataset/DOTA_1024/trainval_split/images/')
    dataset.evaluate(results,None,None,save=False)
    
    # data = []
    # for result,target in results:
    #     img_name = target["filename"]
    #     data.append((result,img_name))

    # dataset.parse_result(data,"test_")



if __name__ == "__main__":
    test_eval()
