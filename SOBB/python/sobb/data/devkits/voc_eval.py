
import numpy as np
from sobb.ops.nms_poly import iou_poly

def parse_gt(filename):
    """
    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]

                if (len(splitlines) == 9):
                    object_struct['difficult'] = 0
                elif (len(splitlines) == 10):
                    object_struct['difficult'] = int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             # cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    # if not os.path.isdir(cachedir):
    #   os.mkdir(cachedir)
    # cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    recs = {}
    for i, imagename in enumerate(imagenames):
        # print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT
        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin = np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]

            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):
                    overlap = iou_poly(BBGT_keep[index], bb)
                    overlaps.append(overlap)
                return overlaps

            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    print('check fp:', fp)
    print('check tp', tp)

    print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def voc_eval_dota(dets,gts,iou_func,ovthresh=0.5,use_07_metric=False):
    dets = np.array(dets.tolist())
    npos = sum([sum(~gts[k]["difficult"]) for k in gts])
    nd = len(dets)
    if nd==0 or npos==0:
        return 0.,0.,0.

    confidence = dets[:,-1]
    dets = dets[:,:-1]

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    scores = confidence[sorted_ind]

    ## note the usage only in numpy not for list
    dets = dets[sorted_ind, :]
    # go down dets and mark TPs and FPs
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d,det in enumerate(dets):
        bb = det[1:].astype(float)
        ovmax = -np.inf
        # S0-9: Handle both integer and string image IDs safely.
        # The image ID is the first column of det; it may be an integer
        # index or a string identifier.
        _img_key = det[0]
        try:
            _img_key = int(_img_key)
        except (ValueError, TypeError):
            _img_key = str(_img_key)
        R = gts[_img_key]
        BBGT = R["box"].astype(float)

        ## compute det bb with each BBGT
        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            BBGT_xmin = np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]

            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):
                    overlap = iou_func(BBGT_keep[index], bb)
                    overlaps.append(overlap)
                return overlaps

            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    # print('check fp:', fp)
    # print('check tp', tp)

    # print('npos num:', npos)
    # print("n dets",nd)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def main():
    detpath = r'test_/{:s}.txt'
    annopath = r'/mnt/disk/lxl/dataset/DOTA_1024/trainval_split/labelTxt/{:s}.txt'  # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    imagesetfile = r'/mnt/disk/lxl/dataset/DOTA_1024/trainval_split/test.txt'

    # For DOTA-v1.5
    # classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #             'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']
    # For DOTA-v1.0
    classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
                  'tennis-court',
                  'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
                  'helicopter']
    classaps = []
    map = 0
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
                                 annopath,
                                 imagesetfile,
                                 classname,
                                 ovthresh=0.5,
                                 use_07_metric=True)
        map = map + ap
        # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print('ap: ', ap)
        classaps.append(ap)

        # # umcomment to show p-r curve of each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.xticks(fontsize=11)
        # plt.yticks(fontsize=11)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # ax = plt.gca()
        # ax.spines['top'].set_color('none')
        # ax.spines['right'].set_color('none')
        # plt.plot(rec, prec)
        # # plt.show()
        # plt.savefig('pr_curve/{}.png'.format(classname))
    map = map / len(classnames)
    print('map:', map)
    classaps = 100 * np.array(classaps)
    print('classaps: ', classaps)


if __name__ == '__main__':
    main()



# ---------------------------------------------------------------------------
# Multi-IoU evaluation: mAP50:95, mAP75, AR, scale-stratified AP.
# Added to support the metrics reported in the manuscript (the legacy
# voc_eval_dota only produced AP50 because the IoU threshold defaulted to 0.5).
# ---------------------------------------------------------------------------
import numpy as np

def _reset_gt_det(classname_gts):
    """Reset GT 'det' flags so each IoU threshold starts fresh."""
    for img_key in classname_gts:
        n = len(classname_gts[img_key]['det'])
        classname_gts[img_key]['det'] = [False] * n


def voc_eval_dota_multi(c_dets, classname_gts, ovthresh_list=None, use_07_metric=False):
    """Evaluate at multiple IoU thresholds with clean GT state per threshold.

    Returns a dict with per-threshold AP, mAP50:95, mAP75, and per-image
    top-k AR (COCO-style) averaged over IoU thresholds.
    """
    if ovthresh_list is None:
        ovthresh_list = np.round(np.arange(0.5, 1.0, 0.05), 2)

    # We need per-image detection counts for AR, so we run a separate
    # per-image AP/AR computation.
    # First, get the image ids and per-image detection counts from c_dets.
    if len(c_dets) == 0:
        return dict(per_iou={}, mAP5095=0.0, mAP75=0.0,
                    AR1=0.0, AR10=0.0, AR100=0.0)

    _img_id_col = c_dets[:, 0]
    try:
        img_ids = _img_id_col.astype(np.int64)
        _use_int_ids = True
    except (ValueError, TypeError):
        img_ids = _img_id_col.astype(str)
        _use_int_ids = False
    unique_imgs = np.unique(img_ids)

    aps = {}
    ar_per_img = {k: [] for k in (1, 10, 100)}

    for ov in ovthresh_list:
        # Reset GT det state before each threshold
        _reset_gt_det(classname_gts)
        rec, prec, ap = voc_eval_dota(c_dets, classname_gts, iou_func=iou_poly,
                                      ovthresh=float(ov), use_07_metric=use_07_metric)
        aps[round(float(ov), 2)] = float(ap)

        # Per-image AR: for each image with GT, compute recall at top-k detections
        # Include images with GT but zero detections (recall = 0)
        for img_id, R in classname_gts.items():
            if R is None:
                continue
            ngt = sum(~R['difficult'])
            if ngt == 0:
                continue
            # Get detections for this image (may be empty)
            _key = int(img_id) if _use_int_ids else str(img_id)
            img_mask = img_ids == _key
            img_dets = c_dets[img_mask] if img_mask.any() else np.array([]).reshape(0, c_dets.shape[1])
            # Handle empty detections
            if len(img_dets) == 0:
                # Zero detections -> zero recall for all k
                for k in (1, 10, 100):
                    ar_per_img[k].append(0.0)
                continue
            # Sort by confidence descending
            scores = img_dets[:, -1]
            order = np.argsort(-scores)
            img_det_polys = img_dets[order, 1:-1]  # polygon coords
            BBGT = R['box'].astype(float)
            if BBGT.size == 0:
                continue
            # Match each detection to GT
            det_matched = np.zeros(len(img_det_polys), dtype=bool)
            gt_matched = np.zeros(BBGT.shape[0], dtype=bool)
            for d_idx in range(len(img_det_polys)):
                bb = img_det_polys[d_idx].astype(float)
                # HBB pre-filter
                BBGT_xmin = np.min(BBGT[:, 0::2], axis=1)
                BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
                BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
                BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
                bb_xmin = np.min(bb[0::2])
                bb_ymin = np.min(bb[1::2])
                bb_xmax = np.max(bb[0::2])
                bb_ymax = np.max(bb[1::2])
                ixmin = np.maximum(BBGT_xmin, bb_xmin)
                iymin = np.maximum(BBGT_ymin, bb_ymin)
                ixmax = np.minimum(BBGT_xmax, bb_xmax)
                iymax = np.minimum(BBGT_ymax, bb_ymax)
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih
                uni = ((bb_xmax-bb_xmin+1.)*(bb_ymax-bb_ymin+1.) +
                       (BBGT_xmax-BBGT_xmin+1.)*(BBGT_ymax-BBGT_ymin+1.) - inters)
                overlaps_hbb = inters / np.maximum(uni, 1e-8)
                keep = overlaps_hbb > 0
                if not keep.any():
                    continue
                best_iou = -1.0
                best_j = -1
                for j in np.where(keep)[0]:
                    if gt_matched[j]:
                        continue
                    iou_val = iou_poly(BBGT[j], bb)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_j = j
                if best_iou > float(ov) and best_j >= 0 and not R['difficult'][best_j]:
                    det_matched[d_idx] = True
                    gt_matched[best_j] = True

            for k in (1, 10, 100):
                top_k_matched = det_matched[:k].sum()
                ar_k = top_k_matched / ngt
                ar_per_img[k].append(float(ar_k))

    mAP5095 = float(np.mean(list(aps.values()))) if aps else 0.0
    mAP75 = aps.get(0.75, 0.0)
    ar1 = float(np.mean(ar_per_img[1])) if ar_per_img[1] else 0.0
    ar10 = float(np.mean(ar_per_img[10])) if ar_per_img[10] else 0.0
    ar100 = float(np.mean(ar_per_img[100])) if ar_per_img[100] else 0.0

    # Reset GT state one final time so subsequent calls see clean state
    _reset_gt_det(classname_gts)

    return dict(per_iou=aps, mAP5095=mAP5095, mAP75=mAP75,
                AR1=ar1, AR10=ar10, AR100=ar100)


def scale_stratified_eval(c_dets, classname_gts, gt_areas, bins=None, ovthresh=0.5):
    """AP broken down by target-area scale bins.

    Parameters
    ----------
    c_dets : np.ndarray [N, 10] (img_id, x1,y1,...,x4,y4, score)
    classname_gts : dict img_id -> {'box': np.ndarray, 'det': list, 'difficult': np.ndarray}
    gt_areas : dict img_id -> np.ndarray of per-GT areas (same order as 'box')
    bins : list of (label, lo, hi) tuples. Defaults to COCO-style small/med/large.
    ovthresh : IoU threshold.

    Returns
    -------
    dict : label -> AP value
    """
    if bins is None:
        bins = [('small', 0, 32**2), ('medium', 32**2, 96**2), ('large', 96**2, 1e12)]
    out = {}
    for label, lo, hi in bins:
        sub_gts = {}
        for img_id, gt_data in classname_gts.items():
            boxes = gt_data['box']
            difficult = gt_data['difficult']
            areas = gt_areas.get(img_id, np.zeros(0))
            if areas.size == 0 or areas.size != len(boxes):
                # Compute areas from polygon coordinates if not provided
                if boxes.ndim == 2 and boxes.shape[1] == 8:
                    xs = boxes[:, 0::2]
                    ys = boxes[:, 1::2]
                    areas = 0.5 * np.abs(
                        (xs[:, 0]*ys[:, 1] - xs[:, 1]*ys[:, 0]) +
                        (xs[:, 1]*ys[:, 2] - xs[:, 2]*ys[:, 1]) +
                        (xs[:, 2]*ys[:, 3] - xs[:, 3]*ys[:, 2]) +
                        (xs[:, 3]*ys[:, 0] - xs[:, 0]*ys[:, 3]))
                else:
                    areas = np.zeros(len(boxes))
            mask = (areas >= lo) & (areas < hi)
            # Keep ALL GTs but mark out-of-bin as difficult (ignored),
            # so detections matching out-of-bin GTs are not false positives.
            if difficult.ndim > 0 and len(difficult) == len(boxes):
                difficult_adj = difficult.copy()
                difficult_adj[~mask] = True
            else:
                difficult_adj = np.where(mask, False, True)
            sub_gts[img_id] = {
                'box': boxes.copy(),
                'det': [False] * len(boxes),
                'difficult': difficult_adj
            }
        # Reset det state before evaluation
        _reset_gt_det(sub_gts)
        _reset_gt_det(classname_gts)
        rec, prec, ap = voc_eval_dota(c_dets, sub_gts, iou_func=iou_poly,
                                     ovthresh=ovthresh, use_07_metric=False)
        _reset_gt_det(sub_gts)
        out[label] = float(ap)
    return out
