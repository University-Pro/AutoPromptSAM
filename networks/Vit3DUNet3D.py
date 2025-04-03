"""
编码器使用ViT3D
解码器使用UNet3D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from torchinfo import summary

# 导入ImageEncoder
from ImageEncoder3D import ImageEncoderViT3D

# 定义3D UNet的编码器部分
class UNet3DEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3DEncoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        pooled = self.pool(x)
        return x, pooled

# 定义3D UNet的解码器部分
class UNet3DDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3DDecoder, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        
        # 编码器部分（保持原样）
        self.encoder = ImageEncoderViT3D(
            img_size=112,
            patch_size=8,
            in_chans=1,
            embed_dim=768,
            depth=12,
            use_abs_pos=False
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            DecoderBlock(256, 128),  # 输出尺寸: 20x28x28
            DecoderBlock(128, 64),   # 输出尺寸: 40x56x56
            DecoderBlock(64, 32)     # 输出尺寸: 80x112x112
        )
        
        # 最终卷积层 + 维度调整
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        # 编码过程
        x = self.encoder(x)
        
        # 解码过程
        x = self.decoder(x)
        
        # 最终卷积
        x = self.final_conv(x)
        
        # 调整维度顺序 (B,C,D,H,W) -> (B,C,H,W,D)
        x = x.permute(0, 1, 3, 4, 2)
        
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device)
    model = UNet3D(in_channels=1, out_channels=2).to(device)
    output = model(input_tensor)
    print("输出形状:", output.shape)  # 应该得到 torch.Size([1, 2, 112, 112, 80])

if __name__ == "__main__":
    # test_PatchEmbed3D()
    # test_patch_merging_3d()
    main()
    # test_Block3D()