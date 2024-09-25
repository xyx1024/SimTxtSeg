'''
Author: xyx1024 8333400+xyx1024@user.noreply.gitee.com
Date: 2023-11-27 09:11:27
LastEditors: silver
LastEditTime: 2024-05-19 04:29:13
FilePath: /pancreas/MyTGSeg/train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '9'
CUDA_LAUNCH_BLOCKING=1
from tqdm import tqdm
import numpy as np
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import VGDataset
from models.model import TxtSimSeg
import argparse
from utils import calculate_metrics


def get_args_parser():
    parser = argparse.ArgumentParser('Segmentation', add_help=False)
    parser.add_argument('--datapath', default='/mnt/data_ssd/yxxie/polyp/', type=str, help='dataset path')
    parser.add_argument('--input_size', default=384, type=int, help='images input size')
    parser.add_argument('--batch_size', default=5, type=int, help='Batch size')
    parser.add_argument('--text_prompt', default='polyp', type=str, help='Text prompt')
    parser.add_argument('--tokenizer',default='bert-base-uncased',type=str,help='tokenizer')
    parser.add_argument('--vision_type',default='facebook/convnext-tiny-224',type=str,help='vision type')
    parser.add_argument('--checkpoint',type=str,help='checkpoint path')
    parser.add_argument('--save_path',type=str,help='result path')
    return parser.parse_args()

def evaluate(model, loader, device,save_path):
    jac = 0.0
    f1 = 0.0
    recall = 0.0
    precision = 0.0
    iou=0.0
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar: 
            for i, ([image,gt],text,name) in enumerate(loader):
                x = image.to(device, dtype=torch.float32) ## 1, 3, 512, 512
                y = gt.to(device, dtype=torch.float32) ##1, 1, 512, 512
                t = {'input_ids':text['input_ids'].to(device, dtype=torch.long),
                'attention_mask':text['attention_mask'].to(device,dtype=torch.long)
                }
                pred = model((x, t))
                mask_pred_show = (pred.squeeze().cpu().numpy())*255
                #print(save_path + name[0])
                #mask_pred_show = F.upsample(mask_pred_show, size=y.shape, mode='bilinear', align_corners=False)
                cv2.imwrite(save_path + name[0], mask_pred_show)
                """ Calculate the metrics """
                batch_jac = []
                batch_f1 = []
                batch_recall = []
                batch_precision = []
                batch_iou=[]
                for y, p in zip(y, pred):
                    score = calculate_metrics(y, p)
                    batch_jac.append(score[0])
                    batch_f1.append(score[1])
                    batch_iou.append(score[2])
                    batch_recall.append(score[3])
                    batch_precision.append(score[4])
                jac += np.mean(batch_jac)
                f1 += np.mean(batch_f1)
                recall += np.mean(batch_recall)
                precision += np.mean(batch_precision)
                iou+=np.mean(batch_iou)
                pbar.update(1)

    jac = jac/len(loader)
    f1 = f1/len(loader)
    iou=iou/len(loader)
    recall = recall/len(loader)
    precision = precision/len(loader)

    return [jac, f1, iou,recall, precision]

if __name__=="__main__":
    args = get_args_parser()
    DATA_DIR = args.datapath
   #dataset_list=["CVC-300","CVC-ClinicDB","CVC-ColonDB","ETIS-LaribPolypDB","Kvasir"]
    x_test_dir = os.path.join(DATA_DIR, 'abnormal/image')
    y_test_dir = os.path.join(DATA_DIR, 'abnormal/mask')
    
    t_size=(args.input_size,args.input_size)

    test_dataset= VGDataset( x_test_dir, y_test_dir, tokenizer=args.tokenizer, t_size=t_size ,text_prompt=args.text_prompt,split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    

    ## test pipeline
    device = torch.device('cuda')
    model = TxtSimSeg(args.tokenizer,args.vision_type)
    #model = torch.nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
    model = model.to(device)
    checkpoint_path = args.checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    save_path=args.save_path
    test_metrics = evaluate(model, test_loader, device,save_path)
    data_str = f"\tTest  Jaccard: {test_metrics[0]:.4f} - F1: {test_metrics[1]:.4f} - IoU: {test_metrics[2]:.4f} - Recall: {test_metrics[3]:.4f} - Precision: {test_metrics[4]:.4f}\n"
    print(data_str)

