# SimTxtSeg: Weakly-Supervised Medical Image Segmentation with Simple Text Cues
Paper : [arxiv](https://arxiv.org/abs/2406.19364),  has been acceptd by ***MICCAI2024✨***

by Yuxin Xie, Tao Zhou, Yi Zhou, Geng Chen



## 🙋 Introduction
Our contribution consists of two key components: an effective Textual-to-Visual Cue Converter that produces visual prompts from text prompts on medical images, and a text-guided segmentation model with Text-Vision Hybrid Attention that fuses text and image features. We evaluate our framework on two medical image segmentation tasks: colonic polyp segmentation and MRI brain tumor segmentation, and achieve consistent state-of-the-art performance.

<img src=images/frame.png width=700 />
<img src=images/attention.png width=700/>

## 🚀 Updates
* `[2024.07.07]` We are excited to release : ✅dataset and ✅TVCC code.
* `[2024.09.25]` We are excited to release : ✅TVHA code.


## 📖 Dataset Preparation
* Dataset Download
    1. Polyp Dataset: [PolypGen](https://www.synapse.org/#!Synapse:syn26376615/wiki/613312) (data_C1 - data_C6 is used), [others](https://github.com/DengPingFan/PraNet) (including CVC-300 (60 samples), CVC-ClinicDB (612 samples), CVC-ColonDB (380 samples), ETIS-LaribPolypDB (196 samples), Kvasir (100 samples), Kvasir-SEG (900 samples))
    2. Brain Tumor Dataset: [kaggle_3m](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)
    3. Isic Dataset: [ISIC](https://challenge.isic-archive.com/data/#2019)
*  For TVCC, to avoid handcrafted prompting cost, <u>we use GPT-4 to generate a concise sentence within 20 words</u>. Before training, you need to transform your dataset into **ODVG** format for precise alignment of regions and phrases. **coco** format label is also required for test and validation.
    ```
    python util/mask2odvg.py
    python util/mask2coco.py
    ```
* For TVHA segmentation model, just use binary mask.

## ⚡ Quick Start
### 1. Environment

Clone the whole repository and install the dependencies.

Python 3.11.9

PyTorch 2.3.1

cuda 12.1
```
conda create -n SimTxtSeg python=3.11
conda activate SimTxtSeg
git clone https://github.com/xyx1024/SimTxtSeg.git
pip install -t requirements.txt
```

see [mmdet_get_started_中文](https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/docs/zh_cn/get_started.md) or [mmdet_get_started_english](https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/docs/en/get_started.md) to install mmdet. 

### 2. For TVCC 
download swin_tiny_patch4_window7_224.pth : https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth

download grounding-dino checkpoints: 
```
wget load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth' # noqa
```
Then use config files to pretrain TVCC：support polyp dataset, brain tumor dataset, isic dataset.
```
cd TVCC/polyp_grounding_dino
./tools/dist_train.sh TVCC/polyp_grounding_dino/config/GroundingDINO_Polyp_PhraseGrounding_config.py n # gpu num, change as you want
```
TVCC evaluation:
```
# 单卡
python tools/test.py config_path ckpt_path

# 4 卡
./tools/dist_test.sh config_path ckpt_path 4
```
visual cues visualize:
```
python tools/image_demo.py 
        image_path \
        config_path \
        --weights weight_path \
        --texts 'xxx'
```
### 3. Pseudo Masks Generation
Click the links below to download the checkpoint for the corresponding model type.

default or vit_h: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

vit_l: [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth).

vit_b: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) & [SAM-Med2d](https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view?usp=drive_link).

Use the checkpoint of SAM and TVCC to generate the pseudo masks.
```
cd TVCC/polyp_grounding_dino
python TVCC_Sam.py
```
### 4. SimTxtSeg with TVHA
use pseudo mask and text prompt to supervise model.
```
python train.py
python test.py
```

## 🎯 Results
**Comparison experiments and Ablation study:**

<img src=images/results.png width=700 />

**Visualization**

<img src=images/visualization.png width=700 />

## 🗓️ Ongoing
- [x] paper release
- [x] dataset release
- [x] TVCC pretrain and test code release
- [x] SimTxtSeg with TVHA model release.

## 🎫 License
This project is released under the Apache 2.0 license.

## 💘 Acknowledge
mmdetection: https://github.com/open-mmlab/mmdetection/tree/main

GroundingDINO: https://github.com/IDEA-Research/GroundingDINO

Segment Anything: https://github.com/facebookresearch/segment-anything?tab=readme-ov-file

## ✒️ Citation
If you find this repository useful, please consider citing this paper:
```
@InProceedings{Xie_SimTxtSeg_MICCAI2024,
        author = { Xie, Yuxin and Zhou, Tao and Zhou, Yi and Chen, Geng},
        title = { { SimTxtSeg: Weakly-Supervised Medical Image Segmentation with Simple Text Cues } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
        year = {2024},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15008},
        month = {October},
        page = {634 -- 644}
}
```
## 📬 Contact
If you have any question, please feel free to contact silver_iris@163.com.
