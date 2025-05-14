"""
DoubeVNet
用VNet检测这种半监督方法是否奏效
"""

import sys
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange
from torchinfo import summary
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Optional, Type, Union, List
import matplotlib.pyplot as plt
from torchinfo import summary

# 导入VNet
from networks.VNet import VNet

class Network(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            image_size: Tuple[int, int, int] = (112, 112, 80),
            patch_size: int = 8,
            embed_dim: int = 768,
            encoder_depth: int = 12,
            out_chans: int = 768,
            num_classes: int = 2,
            normalization: str = "batchnorm",
            has_dropout: bool = True,
            pretrain_weight_path: str = "./result/VNet/LA/Pth/best.pth",
            num_points_per_class: int = 10,
            threshold: float = 0.5,
            mask_in_chans: int = 16,
            activation=nn.GELU,
            num_multimask_outputs: int = 2,
            iou_head_depth: int = 3,
            generatorways: str = "random",
            debug: bool = False,
    ):
        super(Network, self).__init__()

        # ------- 处理粗分割输入参数 -------
        self.in_channels = in_channels
        self.n_classes = num_classes
        self.normalization = normalization
        self.has_dropout = has_dropout

        # ------- ImageEncoderVit3D参数-------
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.out_chans = out_chans

        # ------- PromptGenerator参数 -------
        self.pretrain_weight_path = pretrain_weight_path
        self.num_points_per_class = num_points_per_class
        self.threshold = threshold
        self.generatorways = generatorways
        self.debug = debug

        # ------- PromptEncoder参数 -------
        self.embedding_size = tuple(s // patch_size for s in image_size)  # e.g., 112 -> 14
        self.mask_in_chans = mask_in_chans
        self.activation = activation

        # ------- MaskDecoder参数 -------
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth

        # ------- VNet -------
        self.vnet = VNet(n_channels=self.in_channels,
                         n_classes=self.n_classes,
                         normalization=self.normalization,
                         has_dropout=self.has_dropout)

        self.vnet2 = VNet(n_channels=self.in_channels,
                         n_classes=self.n_classes,
                         normalization=self.normalization,
                         has_dropout=self.has_dropout)

    def forward(self, x):
        # 0.VNet分支1输出结果
        vnet_output = self.vnet(x)

        # 1.VNet分支2输出结果
        vnet_output2 = self.vnet(x)

        return vnet_output,vnet_output2


def networktest():
    # 检查CUDA是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU")

    # 实例化网络
    model = Network(in_channels=1,encoder_depth=4,num_points_per_class=400).to(device=device)

    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device=device)
    vnet_output,sam_output = model(input_tensor)
    print(f"输出形状: {vnet_output.shape,sam_output.shape}")
    summary(model=model,input_size=(1,1,112,112,80))

    return 


if __name__ == "__main__":
    networktest()
    # PromptGenerator_test()