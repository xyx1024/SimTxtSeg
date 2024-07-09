'''
Author: silver
Date: 2024-01-29 06:25:25
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-31 07:50:33
FilePath: /pancreas/MyTGSeg/util/mask2odvg.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import numpy as np
from pycocotools import mask
import cv2
import os
import sys
 
if sys.version_info[0] >= 3:
    unicode = str
 
# TODO:change caption and phrase in need
import io
caption="A polyp is an anomalous oval-shaped small bump-like structure."
phrase="polyp"
tokens_positive=[[2,7]]





# annotations
def maskTobbox(ground_truth_binary_mask):
    contours, _ = cv2.findContours(ground_truth_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 根据二值图找轮廓
    bbox = [] #annotatons in an image
    # print(len(contours),contours)
    global segmentation_id
    if(len(contours)==0):print("0")
    # for each instance
    for i,contour in enumerate(contours):
        if(len(contour)<3):
            print("The contour does not constitute an area")
            continue
        x0, y0, w, h = cv2.boundingRect(contour)
        x1=x0+w
        y1=y0+h
        bbox.append([x0,y0,x1,y1])
    return bbox

# TODO: modify mask path, json path and image path
block_mask_path = '/mnt/data_ssd/yxxie/polyp/Downstream/masks/train'
block_mask_image_files = sorted(os.listdir(block_mask_path))

 
# TODO: json path
jsonPath = "/mnt/data_ssd/yxxie/polyp/Downstream/labels/train_odvg_origin.json"
# TODO: image path, image name should be same as mask name
path = "/mnt/data_ssd/yxxie/polyp/Downstream/images/train"
rgb_image_files = sorted(os.listdir(path))
if block_mask_image_files!=rgb_image_files: print("error")

with io.open(jsonPath, 'w', encoding='utf8') as output:
    # 那就全部写在一个文件夹好了
    # images
    for image_name in rgb_image_files:
        if os.path.exists(os.path.join(block_mask_path, image_name)):
            output.write(unicode('{'))
            image = cv2.imread(os.path.join(path, image_name))
            h,w,_=image.shape
            mask_image=cv2.imread(os.path.join(block_mask_path, image_name),0)
            _, mask_image = cv2.threshold(mask_image, 100, 1, cv2.THRESH_BINARY)
            if not mask_image is None:
                mask_image = np.array(mask_image, dtype=object).astype(np.uint8)
                bbox = maskTobbox(mask_image)
            
            annotation = {
                "filename":image_name,
                "height": h,
                "width": w,
                "grounding":
                {
                    "caption":caption,
                    "regions":[
                        {
                            "bbox":bbox,
                            "phrase":phrase,
                            "tokens_positive":tokens_positive
                        }
                    ]
                }
            }
            str_ = json.dumps(annotation)
            str_ = str_[1:-1]
            if len(str_) > 0:
                output.write(unicode(str_))
                output.write(unicode('}\n'))