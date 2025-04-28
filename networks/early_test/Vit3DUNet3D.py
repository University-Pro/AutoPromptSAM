"""
一号网络使用：
编码器使用ViT3D
解码器使用UNet3D
二号网络使用VNet
一号网络给二号网络提供伪标签
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from torchinfo import summary

# 导入ImageEncoder
from networks.sam_med3d.modeling.image_encoder3D import ImageEncoderViT3D

# 导入VNet网络
from networks.VNet import VNet

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
            depth=4,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            embed_dim=768,
            use_abs_pos=True
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            DecoderBlock(256, 128),  # 输出尺寸: 20x28x28
            DecoderBlock(128, 64),   # 输出尺寸: 40x56x56
            DecoderBlock(64, 32)     # 输出尺寸: 80x112x112
        )
        
        # 最终卷积层 + 维度调整
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

        # 替代depth_adjust层
        self.depth_adjust = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 
                    kernel_size=(3, 1, 1),
                    padding=(1, 0, 0)),
            nn.Upsample(size=(112, 112, 80), mode='trilinear')  # 直接上采样到目标尺寸
        )

        # 创建vnet网络
        self.vnet = VNet(n_channels=1, n_classes=2,normalization="batchnorm", has_dropout=True)

    def padded_intput(self, x):
        # 目标深度  
        target_depth = 112

        # 使用插值填充
        padded_input = F.interpolate(
            x,
            size=(112, 112, target_depth),
            mode='trilinear',
            align_corners=False
        )
        return padded_input

    def padded_output_conv(self, x):
        x = self.depth_adjust(x)
        return x

    def forward(self, x):
        vnet_output = self.vnet(x)

        # 填充输入
        padded_input = self.padded_intput(x)

        # 编码过程
        x = self.encoder(padded_input)
        
        # 解码过程
        x = self.decoder(x)
        
        # 最终卷积
        x = self.final_conv(x)
        
        # 调整维度顺序 (B,C,D,H,W) -> (B,C,H,W,D)
        x = x.permute(0, 1, 3, 4, 2)
        
        # 通过卷积层调整输出形状
        x = self.padded_output_conv(x)

        return x,vnet_output

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device=device)
    print(f'input shape is {input_tensor.shape}')
    model = UNet3D(in_channels=1, out_channels=2).to(device)
    output,vnet_output = model(input_tensor)
    print("output shape is:", output.shape,vnet_output.shape)  # 应该得到 torch.Size([1, 2, 112, 112, 80])

    # 计算模型参数量
    # summary(model=model, input_size=(1, 1, 112, 112, 80), device=device)

if __name__ == "__main__":
    # test_PatchEmbed3D()
    # test_patch_merging_3d()
    main()
    # test_Block3D()