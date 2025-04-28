import medim
import torch
import numpy as np
import torch.nn.functional as F
import torchio as tio
import os.path as osp
from torch import nn
import os
from torchio.data.io import sitk_to_nib
import SimpleITK as sitk


class SAM3D(nn.Module):
    """
    使用SAM3D的Encoder和Decoder实现全监督分割LA数据集
    """

    def __init__(self, create_model_name=None, pretrained=False, checkpoint_path=None, device="CPU"):
        super(SAM3D, self).__init__()

        # 初始化参数
        self.model_name = create_model_name
        self.if_pretrained = pretrained
        self.checkpoint_path = checkpoint_path
        self.device = device

        model = medim.create_model("SAM-Med3D", pretrained=self.if_pretrained, checkpoint_path=self.checkpoint_path)
        print(model)

        # 提取保留patchembed部分
        self.encoder_patchembed = model.image_encoder.patch_embed

        # 提取并保留Encoder的block部分,并且设计模拟输入，测试模拟输出
        self.encoder_blocks = model.image_encoder.blocks
        
        # 提取并且保留neck部分，能够调整输出的通道数
        self.encoder_neck = model.image_encoder.neck

        # 提取decoder部分
        self.decoder = model.mask_decoder
        

    def forward(self, x):
        # # 通过encoder的patchembed部分
        # patchembed_output = self.encoder_patchembed(x)
        # print(f'patchembed_output shape is {patchembed_output.shape}')

        # # 通过encoder的block部分
        # for block in self.encoder_blocks:
        #     patchembed_output = block(patchembed_output)
        #     print(f'block output shape is {patchembed_output.shape}')

        # # 通过encoder的neck部分
        # after_neck_output = self.encoder_neck(patchembed_output.permute(0, 4, 1, 2, 3))
        # print(f'after neck output shape is {after_neck_output.shape}')
        # after_neck_output = after_neck_output.permute(0, 2, 3, 4, 1)
        # print(f'after neck output shape is {after_neck_output.shape}')

        # # 通过decoder部分
        # output = self.decoder(after_neck_output)
        # print(f'output shape is {output.shape}')

        # 设置返回的假数据
        fake_data = torch.randn(1, 1, 112, 112, 80).to(self.device)
        
        return fake_data


if __name__ == "__main__":
    # 生成一个随机的3D数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data = torch.randn(1, 1, 80, 112, 112).to(device)
    # data = torch.randn(1, 1, 192, 192, 192).to(device)  # 这里修改为16的倍数
    data = torch.randn(1, 1, 128, 128, 128).to(device=device)

    # 实例化模型
    model = SAM3D(create_model_name="sam3d", pretrained=True, checkpoint_path="./sam2_configs/sam_med3d_turbo.pth", device=device).to(device)

    # 模型前向传播
    model_output = model(data).to(device=device)