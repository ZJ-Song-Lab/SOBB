import os
import xml.etree.cElementTree as ET
from tqdm import tqdm
import cv2
import numpy as np

def xml2txt(xml_path, txt_path, rescale, annotation_schema='rbox'):
    """Convert SSDD XML annotations to DOTA-style text format.

    Parameters
    ----------
    annotation_schema : str
        'rbox': require <rotated_bndbox>; fail if missing.
        'hbb': use <bndbox> axis-aligned boxes only.
        'auto': try <rotated_bndbox> first, fall back to <bndbox>.
    """
    assert annotation_schema in ('rbox', 'hbb', 'auto'), \
        f"annotation_schema must be 'rbox', 'hbb', or 'auto', got '{annotation_schema}'"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    object=root.findall("object")
    out_lines = []
    n_rbox = 0
    n_hbb = 0
    n_rejected = 0
    for ob in object:
        rbox = ob.find("rotated_bndbox")
        hbox = ob.find("bndbox")
        if annotation_schema == 'rbox':
            if rbox is None:
                n_rejected += 1
                continue
            box = rbox
            x1 = str(float(box.find("x1").text) * rescale[0])
            y1 = str(float(box.find("y1").text) * rescale[1])
            x2 = str(float(box.find("x2").text) * rescale[0])
            y2 = str(float(box.find("y2").text) * rescale[1])
            x3 = str(float(box.find("x3").text) * rescale[0])
            y3 = str(float(box.find("y3").text) * rescale[1])
            x4 = str(float(box.find("x4").text) * rescale[0])
            y4 = str(float(box.find("y4").text) * rescale[1])
            n_rbox += 1
        elif annotation_schema == 'hbb':
            if hbox is None:
                n_rejected += 1
                continue
            box = hbox
            xmin = str(float(box.find("xmin").text) * rescale[0])
            ymin = str(float(box.find("ymin").text) * rescale[1])
            xmax = str(float(box.find("xmax").text) * rescale[0])
            ymax = str(float(box.find("ymax").text) * rescale[1])
            x1, y1, x2, y2, x3, y3, x4, y4 = xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin
            n_hbb += 1
        else:  # auto
            if rbox is not None:
                box = rbox
                x1 = str(float(box.find("x1").text) * rescale[0])
                y1 = str(float(box.find("y1").text) * rescale[1])
                x2 = str(float(box.find("x2").text) * rescale[0])
                y2 = str(float(box.find("y2").text) * rescale[1])
                x3 = str(float(box.find("x3").text) * rescale[0])
                y3 = str(float(box.find("y3").text) * rescale[1])
                x4 = str(float(box.find("x4").text) * rescale[0])
                y4 = str(float(box.find("y4").text) * rescale[1])
                n_rbox += 1
            elif hbox is not None:
                box = hbox
                xmin = str(float(box.find("xmin").text) * rescale[0])
                ymin = str(float(box.find("ymin").text) * rescale[1])
                xmax = str(float(box.find("xmax").text) * rescale[0])
                ymax = str(float(box.find("ymax").text) * rescale[1])
                x1, y1, x2, y2, x3, y3, x4, y4 = xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin
                n_hbb += 1
            else:
                n_rejected += 1
                continue
        name = str(ob.find('name').text)
        diff = ob.find("difficult").text
        data = x1 + " " + y1 + " "+x2 + " " + y2 + " "+x3 + " " + y3 + " "+x4 + " " + y4 + " "+name+' '+diff+"\n"
        out_lines.append(data)

    f = open(txt_path, "w")
    f.writelines(out_lines)
    f.close()
    return {'rbox': n_rbox, 'hbb': n_hbb, 'rejected': n_rejected}

def ssdd_to_dota(img_path, anno_path, target_path, resize, plus, annotation_schema='rbox'):
    names = []
    for root, dirs, files in os.walk(img_path):
        for name in files:
            if not name.endswith(".jpg"):
                continue
            names.append(name[:-4])
    out_img_path = os.path.join(target_path, "images")
    out_anno_path = os.path.join(target_path, "labelTxt")
    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(out_anno_path, exist_ok=True)
    for i in tqdm(range(len(names))):
        name = names[i]
        img = cv2.imread(os.path.join(img_path, name+".jpg"))
        h, w, _ = img.shape
        img = cv2.resize(img, (resize, resize))
        cv2.imwrite(os.path.join(out_img_path, name+".png"), img)
        xml2txt(os.path.join(anno_path, name+'.xml'), os.path.join(out_anno_path, name+'.txt'), (resize / w, resize / h), annotation_schema)