import sys
import os

sys.path.append('/samba/network-storage/ssd/home/pi/sam2-test')  # 设置运行目录

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# 指定GPU设备
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定第0块GPU

# 导入UNet、VNet网络
from network.Unet2D import UNet
from network.VNet import VNet

class UNet2D_VNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet2D_VNet3D, self).__init__()
        self.unet = UNet(n_channels=n_channels, n_classes=n_classes)  # 2D UNet分支
        self.vnet = VNet(n_channels=n_channels, n_classes=n_classes, n_filters=16, has_dropout=True)  # 3D VNet分支

    def slice_3d_to_2d(self, x):
        """将3D张量切片为2D张量列表"""
        slices = []
        for depth in range(x.size(-1)):  # 遍历深度维度
            slices.append(x[:, :, :, :, depth])  # 提取每个切片
        return slices

    def stack_2d_to_3d(self, slices):
        """将2D张量列表重新堆叠为3D张量"""
        stacked = torch.stack(slices, dim=-1)  # 在最后一维堆叠切片
        return stacked

    def forward(self, x):
        # print(f'Input x shape: {x.shape}')
        
        # 使用VNet处理3D张量
        vnet_output = self.vnet(x)  # (1, n_classes, 112, 112, 80)

        # 将3D数据切片为2D张量列表
        slices = self.slice_3d_to_2d(x)  # 调用类方法

        # 逐个将2D切片通过UNet
        processed_slices = []
        for slice_2d in slices:
            processed_slice = self.unet(slice_2d)  # 每个切片通过UNet处理
            processed_slices.append(processed_slice)

        # 将处理后的2D切片重新堆叠为3D张量
        unet_output = self.stack_2d_to_3d(processed_slices)  # 调用类方法

        return vnet_output, unet_output


if __name__ == "__main__":

    # 检查加速功能是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 初始化模型并将其移动到 GPU（如果可用）
    model = UNet2D_VNet3D(n_channels=1, n_classes=2).to(device)
    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device)  # 输入3D张量

    # 前向传播，获取VNet和UNet的输出
    output_vnet, output_unet = model(input_tensor)
    print("VNet Output shape:", output_vnet.shape)
    print("UNet Output shape:", output_unet.shape)
