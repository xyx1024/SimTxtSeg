import json
import numpy as np
from pycocotools import mask
import cv2
import os
import sys
 
if sys.version_info[0] >= 3:
    unicode = str
 
 
import io
global segmentation_id
segmentation_id = 1
# annotations
def maskToanno(ground_truth_binary_mask, ann_count, category_id):
    contours, _ = cv2.findContours(ground_truth_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 根据二值图找轮廓
    annotations = [] #annotatons in an image
    # print(len(contours),contours)
    global segmentation_id
    if(len(contours)==0):print("0")
    # for each instance
    for i,contour in enumerate(contours):
        if(len(contour)<3):
            print("The contour does not constitute an area")
            continue
        ground_truth_area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        annotation = {
            "segmentation": [],
            "area": ground_truth_area,
            "iscrowd": 0,
            "image_id": ann_count,
            "bbox": [x,y,w,h],
            "category_id": category_id,
            "id": segmentation_id
        }
        # segmentation
        contour = np.flip(contour, axis=0)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)
        annotations.append(annotation)
        segmentation_id = segmentation_id + 1
    return annotations
 
# TODO: maskpath
block_mask_path = '...'
block_mask_image_files = sorted(os.listdir(block_mask_path))
 
 
# TODO: json path
jsonPath = "..."
annCount = 1
imageCount = 1
# TODO: image path, image name should be same as mask name
path = "..."
rgb_image_files = sorted(os.listdir(path))
if block_mask_image_files!=rgb_image_files: print("error")
 
with io.open(jsonPath, 'w', encoding='utf8') as output:
    output.write(unicode('{\n'))
    # information
    output.write(unicode('"info": [\n'))
    output.write(unicode('{\n'))
    info={
        "year": "2023",
        "version": "1",
        "contributor": "bulibuli",
        "url": "",
        "date_created": "2023-01-17"
    }
    str_ = json.dumps(info, indent=4)
    str_ = str_[1:-1]
    if len(str_) > 0:
        output.write(unicode(str_))
    output.write(unicode('}\n'))
    output.write(unicode('],\n'))
 
    #lisence
    output.write(unicode('"lisence": [\n'))
    output.write(unicode('{\n'))
    info={
        "id": 1,
        "url": "https://creativecommons.org/licenses/by/4.0/",
        "name": "CC BY 4.0"
    }
    str_ = json.dumps(info, indent=4)
    str_ = str_[1:-1]
    if len(str_) > 0:
        output.write(unicode(str_))
    output.write(unicode('}\n'))
    output.write(unicode('],\n'))
 
    # category
    output.write(unicode('"categories": [\n'))
    output.write(unicode('{\n'))
    categories = {
        "supercategory": "polyp",
        "id": 1,
        "name": "polyp"
    }
    str_ = json.dumps(categories, indent=4)
    str_ = str_[1:-1]
    if len(str_) > 0:
        output.write(unicode(str_))
    output.write(unicode('}\n'))
    output.write(unicode('],\n'))
 
    # images
    output.write(unicode('"images": [\n'))
    for image in rgb_image_files:
        if os.path.exists(os.path.join(block_mask_path, image)):
            output.write(unicode('{'))
            block_im = cv2.imread(os.path.join(path, image))
            h,w,_=block_im.shape
            annotation = {
                "height": h,
                "width": w,
                "id": imageCount,
                "file_name": image
            }
            str_ = json.dumps(annotation, indent=4)
            str_ = str_[1:-1]
            if len(str_) > 0:
                output.write(unicode(str_))
                imageCount = imageCount + 1
            if (image == rgb_image_files[-1]):
                output.write(unicode('}\n'))
            else:
                output.write(unicode('},\n'))
    output.write(unicode('],\n'))
 
    # annotations
    output.write(unicode('"annotations": [\n'))
    for i in range(len(block_mask_image_files)):
        if os.path.exists(os.path.join(path, block_mask_image_files[i])):
            block_image = block_mask_image_files[i]
            # print(block_image)
            # binary mode
            block_im = cv2.imread(os.path.join(block_mask_path, block_image), 0)
            _, block_im = cv2.threshold(block_im, 100, 1, cv2.THRESH_BINARY)
            if not block_im is None:
                block_im = np.array(block_im, dtype=object).astype(np.uint8)
                block_anno = maskToanno(block_im, annCount, 1)
                # print(block_image,len(block_anno))
                for b in block_anno:
                    str_block = json.dumps(b, indent=4)
                    str_block = str_block[1:-1]
                    if len(str_block) > 0:
                        output.write(unicode('{\n'))
                        output.write(unicode(str_block))
                        if (block_image == rgb_image_files[-1] and b == block_anno[-1]):
                            output.write(unicode('}\n'))
                        else:
                            output.write(unicode('},\n'))
                annCount = annCount + 1
            else:
                print(block_image)
            
    output.write(unicode(']\n'))
    output.write(unicode('}\n'))
 
 