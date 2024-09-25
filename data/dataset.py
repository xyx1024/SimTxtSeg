'''
Author: xyx1024 8333400+xyx1024@user.noreply.gitee.com
Date: 2024-02-05 10:32:41
LastEditors: silver
LastEditTime: 2024-02-26 10:43:43
FilePath: /pancreas/MyTGSeg/data/dataset.py
Description: visual grounding dataset
'''
from typing import Any
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import random
from transformers import AutoTokenizer
import albumentations as A

class VGDataset(Dataset):
    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 text_prompt: str,
                 tokenizer,
                 t_size,
                 split
                 ) -> None:
        super().__init__()
        self.image_list=[os.path.join(image_dir,image_name) for image_name in sorted(os.listdir(image_dir))]
        self.mask_list=[os.path.join(mask_dir,mask_name) for mask_name in sorted(os.listdir(mask_dir))]
        self.text_prompt=len(self.image_list)*[text_prompt]
        self.t_size=t_size
        self.split=split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        self.transform= A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32),
    ])

    
    def __getitem__(self, index: int) -> Any:
        image=cv2.imread(self.image_list[index],cv2.IMREAD_COLOR)
        name=self.image_list[index].split('/')[-1]
        gt=cv2.imread(self.mask_list[index],cv2.IMREAD_GRAYSCALE)
        text_prompt=self.text_prompt[index]
        if self.transform is not None and self.split == 'train':
            augmentations=self.transform(image=image,mask=gt)
            image=augmentations['image']
            gt=augmentations['mask']

        """ Image """
        image = cv2.resize(image, self.t_size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0

        """ Mask """
        gt = cv2.resize(gt, self.t_size)
        gt = np.expand_dims(gt, axis=0)
        gt = gt/255.0
        
        token_output = self.tokenizer.encode_plus(text_prompt, padding='max_length',
                                                        max_length=24, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token,mask = token_output['input_ids'],token_output['attention_mask']
        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)} 
        if self.split == 'test':
            return ([image,gt],text,name)
        return ([image,gt],text)
    
    def __len__(self):
        return len(self.image_list)


if __name__=="__main__":

    dataset_vg = VGDataset("/mnt/data_ssd/yxxie/polyp/Downstream/images/train",
                           "/mnt/data_ssd/yxxie/polyp/Downstream/masks/train",
                           "polyp .",
                           "bert-base-uncased",
                           (512,512))

    print(len(dataset_vg))
    data = dataset_vg[random.randint(0, 10)] 
    print(data)



