'''
Descripttion: your project
version: 1.0
Author: silver
Date: 2024-02-20 07:47:51
LastEditors: xyx1024 8333400+xyx1024@user.noreply.gitee.com
LastEditTime: 2024-09-24 13:21:01
'''
import torch
import torch.nn as nn
from einops import rearrange
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, dropout=0, max_len:int=500000) -> None:

        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)
        self.register_buffer('pe', pe)  

    def forward(self, x):

        #  output = word_embedding + positional_embedding
        x = x + nn.Parameter(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]


class LangFusionLayer(nn.Module):

    def __init__(self, in_channels:int, output_text_len:int, spatial_size:int, input_text_len:int=24, embed_dim:int=768):

        super(LangFusionLayer, self).__init__()

        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm1 = nn.LayerNorm(in_channels)
        
        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=1,batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)

        self.cross_attn1 = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)
        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len,output_text_len,kernel_size=1,stride=1),
            nn.GELU(),
            nn.Linear(embed_dim,in_channels),
            nn.LeakyReLU(),
        )

        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels,max_len=output_text_len)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.scale = nn.Parameter(torch.tensor(0.01),requires_grad=True)


    def forward(self,x,txt):

        '''
        x:[B N C1]  
        txt:[B,L,C]
        '''
        x = rearrange(x,'b c h w -> b (h w) c')

        txt = self.text_project(txt)# [B L C1]  

       # Self-Attention
        vis2 = self.norm1(x) #[B N C1]  
        q = k = self.vis_pos(vis2)
        vis2 = self.self_attn(q, k, value=vis2)[0] #[B N C1]  
        vis2 = self.self_attn_norm(vis2) #[B N C1]  
        vis = x + vis2 #[B N C1]  

        #TXT-VIS Cross-Attention
        txt2,_=self.cross_attn1(query=self.txt_pos(txt),
                               key=self.vis_pos(vis),
                               value=vis)
        txt2=self.cross_attn_norm1(txt2)
        txt=txt+txt2

        # VIS-TXT Cross-Attention
        vis2 = self.norm2(vis)
        vis2,_ = self.cross_attn(query=self.vis_pos(vis2),
                                   key=self.txt_pos(txt),
                                   value=txt) #[B N C1]  
        vis2 = self.cross_attn_norm(vis2) #[B N C1]  
        vis = vis + self.scale*vis2 #[B N C1]  

        vis = rearrange(vis, 'B (H W) C -> B C H W',H=self.spatial_size,W=self.spatial_size) #[1,32,128,128]

        return vis