from torch import nn
import torch
import numpy as np
import math
import torch.nn.functional as F

class add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        rs = args[0]
        for i in range(1, len(args)):
            rs = rs + args[i]
        return rs

class concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        rs = args[0]
        for i in range(1, len(args)):
            rs = torch.cat((rs, args[i]), dim=1)
        return rs
    


def get_module(t,layer_params,module_list):
    if t == "conv1d":
        layer = nn.Conv1d(**layer_params)
    elif t == "conv2d":
        layer = nn.Conv2d(**layer_params)
    elif t == "conv3d":
        layer = nn.Conv3d(**layer_params)
    elif t == "linear":
        layer = nn.Linear(**layer_params)
    elif t == "add":
        # return "ADD_OP"
        layer = add()
        return layer, True
    elif t == "concat":
        # return "CONCAT_OP"
        layer = concat()
        return layer, True
    elif t == "flatten":
        return nn.Flatten(), True
    elif t == "adaptiveavgpool1d":
        return nn.AdaptiveAvgPool1d(**layer_params), True
    elif t == "adaptiveavgpool2d":
        return nn.AdaptiveAvgPool2d(**layer_params), True
    elif t == "adaptiveavgpool3d":
        return nn.AdaptiveAvgPool3d(**layer_params), True
    elif t == "dropout":
        return nn.Dropout(**layer_params), True
    elif t == 'self_attn_with_ffa':
        return SelfAttentionWithFFA(**layer_params), True
    else:
        raise ValueError(f"Unsupported type {t}")
    
    module_list.append(layer)
    return layer, False

def get_activation_bn(act, bn, layer, module_list):
    if bn:
        if bn == "batchnorm1d":
            bn_layer = nn.BatchNorm1d(layer.out_channels)
        elif bn == "batchnorm2d":
            bn_layer = nn.BatchNorm2d(layer.out_channels)
        elif bn == "batchnorm3d":
            bn_layer = nn.BatchNorm3d(layer.out_channels)
        else:
            raise ValueError(f"Unsupported bn type {bn}")
        module_list.append(bn_layer)
    if act:
        act_layer = nn.ReLU() if act == "relu" else nn.Sigmoid()
        module_list.append(act_layer)



class FeedForwardAttention(nn.Module):
    def __init__(self, in_channels,out_channels, hidden_channels=None, dropout=0.1):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.attn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),   # 通道注意力得分
            nn.Sigmoid()
        )
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        h = F.relu(self.fc1(x))              # (B, L, hidden)
        attn_weights = self.attn(h)          # (B, L, 1)
        h = h * attn_weights                 # 加权特征（token-level attention）
        h = self.dropout(self.fc2(h))        # (B, L, D)
        return h

class SelfAttentionWithFFA(nn.Module):
    def __init__(self, in_channels, out_channels=256,hidden_channels=None, heads=4, dropout=0.1):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffa = FeedForwardAttention(in_channels, out_channels, hidden_channels ,dropout=dropout)

    def forward(self, x):
        """
        x: (B, C, L)
        x: (B, C, H, W)
        x: (B, C, D, H, W)
        """
        if x.dim() == 5:
            dimension = '3d'
            B, C, D, H, W = x.shape
            x = x.flatten(2).permute(0, 2, 1)  # (B, L, C)
        elif x.dim() == 4:
            dimension = '2d'
            B, C, H, W = x.shape
            x = x.flatten(2).permute(0, 2, 1)  # (B, L, C)
        else:
            dimension = '1d'
            B, C, L = x.shape
            x = x.permute(0, 2, 1)  # (B, L, C)

        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # q=k=v=x
        x = x + attn_out  # residual

        # FeedForward + Attention (FFA)
        x_norm2 = self.norm2(x)
        ffa_out = self.ffa(x_norm2)
        x = x + ffa_out  # residual

        if dimension == '3d':
            x = x.permute(0, 2, 1).view(B, C, D, H, W)
        elif dimension == '2d':
            x = x.permute(0, 2, 1).view(B, C, H, W)
        else:
            x = x.permute(0, 2, 1)

        return x
