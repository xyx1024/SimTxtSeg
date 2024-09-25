'''
Author: xyx1024 8333400+xyx1024@user.noreply.gitee.com
Date: 2023-11-27 09:11:27
LastEditors: silver
LastEditTime: 2024-09-25 09:24:06
FilePath: /pancreas/MyTGSeg/train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA_LAUNCH_BLOCKING=1
from tqdm import tqdm
import numpy as np
import torch
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import VGDataset
import argparse
from models.model import TxtSimSeg
from monai.losses import DiceCELoss
import time
from utils import create_dir,print_and_log,calculate_metrics,epoch_time


def get_args_parser():
    parser = argparse.ArgumentParser('Segmentation', add_help=False)
    parser.add_argument('--datapath', default='/mnt/data_ssd/yxxie/polyp/', type=str, help='dataset path')
    parser.add_argument('--input_size', default=384, type=int, help='images input size')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--text_prompt', default='polyps xxxx .', type=str, help='Text prompt')
    parser.add_argument('--tokenizer',default='bert-base-uncased',type=str,help='tokenizer')
    parser.add_argument('--vision_type',default='facebook/convnext-tiny-224',type=str,help='vision type')
    parser.add_argument('--checkpoint',default='/mnt/data_ssd/yxxie/polyp/SimTxtSeg/CKPT/',type=str,help='checkpoint path')
    return parser.parse_args()

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0
    
    model.train()
    with tqdm(total=len(loader)) as pbar: 
        for i, ([image,gt],text) in enumerate(loader):
            x = image.to(device, dtype=torch.float32) 
            y = gt.to(device, dtype=torch.float32) 
            t = {'input_ids':text['input_ids'].to(device, dtype=torch.long),
            'attention_mask':text['attention_mask'].to(device,dtype=torch.long)
            }
            input=(x, t)
            optimizer.zero_grad()
            pred = model(input)
            loss=loss_fn(pred,y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []
            for y, p in zip(y, pred):
                score = calculate_metrics(y, p)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])
            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)
            pbar.update(1)

    epoch_loss = epoch_loss/len(loader)
    epoch_jac = epoch_jac/len(loader)
    epoch_f1 = epoch_f1/len(loader)
    epoch_recall = epoch_recall/len(loader)
    epoch_precision = epoch_precision/len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar: 
            for i, ([image,gt],text) in enumerate(loader):
                x = image.to(device, dtype=torch.float32) 
                y = gt.to(device, dtype=torch.float32) 
                t = {'input_ids':text['input_ids'].to(device, dtype=torch.long),
                'attention_mask':text['attention_mask'].to(device,dtype=torch.long)
                }
                pred = model((x, t))
                loss=loss_fn(pred,y)
                epoch_loss += loss.item()

                """ Calculate the metrics """
                batch_jac = []
                batch_f1 = []
                batch_recall = []
                batch_precision = []
                for y, p in zip(y, pred):
                    score = calculate_metrics(y, p)
                    batch_jac.append(score[0])
                    batch_f1.append(score[1])
                    batch_recall.append(score[2])
                    batch_precision.append(score[3])
                epoch_jac += np.mean(batch_jac)
                epoch_f1 += np.mean(batch_f1)
                epoch_recall += np.mean(batch_recall)
                epoch_precision += np.mean(batch_precision)
                pbar.update(1)

    epoch_loss = epoch_loss/len(loader)
    epoch_jac = epoch_jac/len(loader)
    epoch_f1 = epoch_f1/len(loader)
    epoch_recall = epoch_recall/len(loader)
    epoch_precision = epoch_precision/len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

if __name__=="__main__":
    args = get_args_parser()
    DATA_DIR = args.datapath
   
    x_train_dir = os.path.join(DATA_DIR, 'images/train/')
    x_valid_dir = os.path.join(DATA_DIR, 'images/val/')
    y_train_dir = os.path.join(DATA_DIR, 'fake_masks/train/')
    y_valid_dir = os.path.join(DATA_DIR, 'masks/val/')
    
    t_size=(args.input_size,args.input_size)

    train_dataset = VGDataset( x_train_dir, y_train_dir, tokenizer=args.tokenizer, t_size=t_size,text_prompt=args.text_prompt,split='train' )
    valid_dataset = VGDataset( x_valid_dir, y_valid_dir, tokenizer=args.tokenizer, t_size=t_size ,text_prompt=args.text_prompt,split='train')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    create_dir("outputs")
    train_log_path = "outputs/train_log_brain.txt"
    train_log = open(train_log_path, "w")
    train_log.write("\n")
    train_log.close()

    ## train pipeline
    device = torch.device('cuda')
    model = TxtSimSeg(args.tokenizer,args.vision_type)
    #model = torch.nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_log(train_log_path, data_str)
    num_epochs=100
    best_valid_metrics = 0.0
    early_stopping_count = 0
    early_stopping_patience = 50


    for epoch in range(num_epochs):
        start_time = time.time()
        print("epoch",epoch)
        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[1] > best_valid_metrics:
            data_str = f"Valid F1 improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint: {args.checkpoint}"
            print_and_log(train_log_path, data_str)

            best_valid_metrics = valid_metrics[1]
            checkpoint_path=args.checkpoint+"convnext_dice_"+str(best_valid_metrics)
            torch.save(model.state_dict(),checkpoint_path )
            early_stopping_count = 0

        elif valid_metrics[1] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_log(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_log(train_log_path, data_str)
            break

