import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from argparse import Namespace
from collections import defaultdict

# mmdet
try:
    import mmdet
    from mmdet.apis import inference_detector, init_detector
except ImportError:
    mmdet = None

import sys
sys.path.append('../')
from mmengine.config import Config
from mmengine.utils import ProgressBar
from PIL import Image
from tools.utils import get_file_list

# grounding dino
from mmdet.apis.det_inferencer import DetInferencer
import ast

"""
Write In Before
We use DetInferencer in our program, while edit the "call function" in this package
Please find call function in DetInference, and rewrite that "for" loop:
>>>
for ori_imgs, data in (track(inputs, description='Inference')
                               if self.show_progress else inputs):
            preds = self.forward(data, **forward_kwargs)
            return preds
>>>
Just comment out the code related to the visualization and results variables.
We only need the preds.
"""

def parse_args():
    parser = argparse.ArgumentParser(
        'Detect-Segment-Anything Demo', add_help=True)
    parser.add_argument('--image', type=str, help='path to image file')
    parser.add_argument('--det_config', type=str, 
        default='xxxx',help='path to det config file')
    parser.add_argument('--det_weight', type=str,
        default='xxxx',help='path to det weight file')
    
    parser.add_argument(
        '--sam-type',type=str,
        default='vit_h',choices=['vit_h', 'vit_l', 'vit_b'],help='sam type')
    parser.add_argument(
        '--sam-weight',type=str,
        default='xxxx',help='path to checkpoint file')
    parser.add_argument(
        '--out-dir','-o',type=str,help='output directory')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--box-thr', '-b', type=float, 
        default=0.25, help='box threshold')
    parser.add_argument(
        '--det-device','-d',
        default='cuda:4',help='Device used for inference')
    parser.add_argument(
        '--sam-device','-s',
        default='cuda:4',help='Device used for inference')
    parser.add_argument(
        '--text-prompt', '-t', type=str, 
        default='A nuclear magnetic resonance image of brain tumors.',help='text prompt')
    parser.add_argument(
        '--text-thr', type=float, default=0.25, help='text threshold')
    parser.add_argument(
        '--apply-original-groudingdino',action='store_true',
        default=False,help='use original groudingdino label predict')


    # only for Grounding DINO
    parser.add_argument(
        '--custom-entities','-c', action='store_true',
        default=False,
        help='Whether to customize entity names? '
        'If so, the input text should be '
        '"cls_name1 . cls_name2 . cls_name3 ." format')
    
    parser.add_argument(
        '--chunked-size',type=int,
        default=-1,help='If the number of categories is very large, '
        'you can specify this parameter to truncate multiple predictions.')
    
    parser.add_argument(
        '--tokens-positive','-p',type=str,
        default="[[[38,49]]]",
        help='Used to specify which locations in the input text are of '
        'interest to the user. -1 indicates that no area is of interest, '
        'None indicates ignoring this parameter. '
        'The two-dimensional array represents the start and end positions.')
    
    return parser.parse_args()

def build_init_args(args):
    init_args={}
    init_args['show_progress']=False
    init_args['model']=args.det_config
    init_args['weights']=args.det_weight
    init_args['device']=args.det_device
    init_args['palette']=None
    return init_args

def build_call_args(args,image_path):
    call_args={}
    call_args['inputs']=image_path
    call_args['texts']=args.text_prompt
    call_args['custom_entities']=args.custom_entities
    call_args['pred_score_thr']=args.box_thr
    if args.tokens_positive is not None:
        call_args['tokens_positive'] = ast.literal_eval(
            args.tokens_positive)
    return call_args
 
def build_detecter(args):
    config = Config.fromfile(args.det_config)
    detecter = init_detector(
        config, args.det_weight, device=args.det_device, cfg_options={})
    return detecter
def build_sam(args):
    if "med2d" not in args.sam_weight:
        from segment_anything import SamPredictor, sam_model_registry
        sam_model = SamPredictor(sam_model_registry[args.sam_type](checkpoint=args.sam_weight))
    else:
        from segment_anything_med2d import SamPredictor, sam_model_registry
        sam_args = Namespace()
        sam_args.image_size = 256
        sam_args.encoder_adapter = True
        sam_args.sam_checkpoint=args.sam_weight
        sam_model = SamPredictor(sam_model_registry[args.sam_type](sam_args))
    sam_model.mode = sam_model.model.to(args.sam_device)
    return sam_model
def run_detector(model,image_path,args):
    pred_dict = {}
    text_prompt = args.text_prompt
    text_prompt = text_prompt.lower()
    text_prompt = text_prompt.strip()
    if not text_prompt.endswith('.'):
        text_prompt = text_prompt + '.'
    if args.apply_original_groudingdino:
        result = inference_detector(model, image_path,text_prompt=text_prompt)
    else:
        call_args=build_call_args(args,image_path)
        result=model(**call_args)[0]
    pred_instances = result.pred_instances[
        result.pred_instances.scores > args.box_thr]
    #print(pred_instances)
    pred_dict['boxes'] = pred_instances.bboxes
    pred_dict['scores'] = pred_instances.scores.cpu().numpy().tolist()
    pred_dict['labels'] = [
        label
        # model.dataset_meta['classes'][label]
        for label in pred_instances.labels
    ]
    
    # if args.use_detic_mask:
    #     pred_dict['masks'] = pred_instances.masks
    return pred_dict

def draw_and_save(image,
                  pred_dict,
                  save_path,
                  random_color=True,
                  show_label=True):
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    with_mask = 'masks' in pred_dict
    labels = pred_dict['labels']
    scores = pred_dict['scores']
    bboxes = pred_dict['boxes'].cpu().numpy()

    for box, label, score in zip(bboxes, labels, scores):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        plt.gca().add_patch(
            plt.Rectangle((x0, y0),w,h,edgecolor='green',
                          facecolor=(0, 0, 0, 0),lw=2))
        if show_label:
            if isinstance(score, str):
                plt.gca().text(x0, y0, f'{label}|{score}', color='white')
            else:
                plt.gca().text(
                    x0, y0, f'{label}|{round(score,2)}', color='white')
    if with_mask:
        masks = pred_dict['masks'].cpu().numpy()
        for mask in masks:
            if random_color:
                color = np.concatenate(
                    [np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            plt.gca().imshow(mask_image)
    plt.axis('off')
    plt.savefig(save_path)
    
def save_fake_mask(masks,path):
    masks_np=[mask.squeeze().cpu().numpy() for mask in masks]
    for i, mask_np in enumerate(masks_np):
        mask_path = path+f"_mask_{i+1}.png"
        mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_image.save(mask_path)

##  instance masks fusion
def make_pair(dataset_dir):     
    #list
    pair_dict = defaultdict(list)
    for image_name in os.listdir(dataset_dir):
        image=image_name.split(".")[0]
        pair_dict[image].append(os.path.join(dataset_dir,image_name))
    return pair_dict

def mask_add(image_mask_pair):
    for value in image_mask_pair.values():
        old=value[0]
        new=old.split('.')[0]+".png"
        print(value)
        if len(value)==1: 
            os.rename(old,new)
        else:
            result=cv2.imread(value[0], cv2.IMREAD_GRAYSCALE)
            os.remove(value[0])
            for i in range(1,len(value)):
                mask = cv2.imread(value[i], cv2.IMREAD_GRAYSCALE)
                result = cv2.add(result, mask)
                os.remove(value[i])
            cv2.imwrite(new, result)

def main():
    args = parse_args()
    out_dir = args.out_dir
    if args.apply_original_groudingdino:
        det_model = build_detecter(args)
    else:
        init_args=build_init_args(args)
        det_model=DetInferencer(**init_args)
        chunked_size=args.chunked_size
        det_model.model.test_cfg.chunked_size = chunked_size

    # print(det_model)
    sam_model=build_sam(args)

    os.makedirs(out_dir, exist_ok=True)
    files, source_type = get_file_list(args.image)
    progress_bar = ProgressBar(len(files))

    # generate instance masks
    for image_path in files:
        #print(image_path)
        save_path = os.path.join(out_dir,"test",os.path.basename(image_path))
        #save_path2 = os.path.join(out_dir,"test2",os.path.basename(image_path))
        pred_dict=run_detector(det_model,image_path,args)
        box=pred_dict['boxes']

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam_model.set_image(image)
        transformed_boxes = sam_model.transform.apply_boxes_torch(
                box, image.shape[:2])
        transformed_boxes = transformed_boxes.to(sam_model.model.device)
        #print(transformed_boxes)
        if transformed_boxes.shape[0]==0:
            h,w,_=image.shape
            mask_np = np.zeros((h, w), dtype = np.uint8)
            mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_image.save(os.path.join(save_path))
        else:
            masks, _, _ = sam_model.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False)
            pred_dict['masks'] = masks
            save_fake_mask(masks,save_path)
            # draw_and_save(image,pred_dict,save_path2)
        progress_bar.update()
    
    out_dir=os.path.join(out_dir,"test")
    image_mask_pair=make_pair(out_dir)
    mask_add(image_mask_pair)  

if __name__ == '__main__':
    main()