'''
Author: xyx1024 8333400+xyx1024@user.noreply.gitee.com
Date: 2024-02-07 11:46:36
LastEditors: silver
LastEditTime: 2024-09-25 10:24:55
FilePath: /pancreas/MyTGSeg/models/model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
from transformers import  AutoModel
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample


from models.layers.fusion_layer import LangFusionLayer
from models.layers.modules import  UpCat
from models.layers.channel_attention_layer import SE_Conv_Block
class TextualModel(nn.Module):

    def __init__(self, bert_type):

        super(TextualModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        # freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        return output['hidden_states']

class VisualModel(nn.Module):
    def __init__(self,vision_type) -> None:
        super(VisualModel,self).__init__()
        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True,trust_remote_code=True)


    def forward(self, x):
        output = self.model(x, output_hidden_states=True)
        return output['hidden_states']
    
class Comprehensive_Atten_Decoder(nn.Module):
    def __init__(self, nonlocal_mode='concatenation', attention_dsample=(1, 1),is_deconv=True) -> None:
        super(Comprehensive_Atten_Decoder,self).__init__()
        filters=[96,192,384,768]
        text_len = [24,12,9]
        
        spatial_dim = [12,24,48,96]
        self.is_deconv = is_deconv
       
        # upsampling
        self.up_concat4 = UpCat(filters[3], filters[2], self.is_deconv)
        self.up_concat3 = UpCat(filters[2], filters[1], self.is_deconv)
        self.up_concat2 = UpCat(filters[1], filters[0], self.is_deconv)
        self.up_concat1 = UpCat(filters[1], filters[0], self.is_deconv)
        self.up4 = SE_Conv_Block(filters[3], filters[2], drop_out=True)
        self.up3 = SE_Conv_Block(filters[2], filters[1])
        self.up2 = SE_Conv_Block(filters[1], filters[0])
        self.up1 = SubpixelUpsample(2,filters[0],24,4)
    
        #fusion layer
        self.fusion1 = LangFusionLayer(filters[3],text_len[0],spatial_dim[1])
        self.fusion2 = LangFusionLayer(filters[2],text_len[1],spatial_dim[2])
        self.fusion3 = LangFusionLayer(filters[1],text_len[2],spatial_dim[3])

        self.project = nn.Sequential(
            nn.Conv1d(filters[3],filters[2],kernel_size=1,stride=1),
            nn.GELU()
        )
        

    def forward(self,vis_embeds, txt_embeds):
        # [b,96,96,96] [b,192,48,48] [b,384,24,24] [b,768,12,12]
        up4 = self.up_concat4(vis_embeds[2],vis_embeds[3])# [b,768,24,24]
        up4=self.fusion1(up4,txt_embeds)#[b,768,24,24]
        up4, att_weight4 = self.up4(up4)#[b,384,24,24]

        up3 = self.up_concat3(vis_embeds[1], up4)#[b,384,48,48]
        up3=self.fusion2(up3,txt_embeds)#[b,384,48,48]
        up3, att_weight3 = self.up3(up3)#[b,192,48,48]

        up2 = self.up_concat2(vis_embeds[0], up3)
        up2=self.fusion3(up2,txt_embeds)
        up2, att_weight2 = self.up2(up2)

        up1 = self.up1(up2) 
        return up1
        

class TxtSimSeg(nn.Module):
    def __init__(self, bert_type,vision_type):

        super(TxtSimSeg, self).__init__()
        self.text_encoder = TextualModel(bert_type)
        self.vision_encoder = VisualModel(vision_type)
        self.spatial_dim = [12,24,48,96]

        self.decoder=Comprehensive_Atten_Decoder()
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):
        image, text = data # [B,C,H,W]
        image_embeds = self.vision_encoder(image)
        image_embeds = image_embeds[1:]# [48,96,96,96] [48,192,48,48] [48,384,24,24] [48,768,12,12]
        text_embeds = self.text_encoder(text['input_ids'],text['attention_mask'])# 13x[48 24 768]
        os1=self.decoder(image_embeds,text_embeds[-1])
        out = self.out(os1).sigmoid()

        return out
        
