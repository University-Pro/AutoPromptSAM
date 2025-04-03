"""
SAM2生成伪标签
SSL
VNet处理3D数据
"""
import sys
sys.path.append('/samba/network-storage/ssd/home/pi/sam2-test')  # 设置运行目录

from torchinfo import summary
from einops import rearrange
import torch.nn.functional as F
import math

# 导入SAM2
from sam2.build_sam import build_sam2

import torch.nn as nn
import torch

# 导入VNet网络
from VNet import VNet

class Final_PatchExpand2D(nn.Module):
    """
    最终的四倍放大
    通道数变成原来的四分之一
    """
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x = self.norm(x)

        return x

# 双层卷积
class DoubleConv(nn.Module):
    """
    来自UNet的双层卷积
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# 上采样部分
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels=out_channels *
                               2, out_channels=out_channels)

    # 设计跳连，其中x2是跳连传递进来的
    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 适配尺寸
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 合并上采样的特征图和跳跃连接的特征图
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 定义最终的输出卷积层
class OutConv(nn.Module):
    """
    最后的输出卷积层，将特征图的通道数映射到目标类别数。
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # 1x1卷积，将通道数从in_channels映射到out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SAM2VNet(nn.Module):
    """
    使用SAM2的Encoder和VNet实现SSL半监督网络
    """

    def __init__(self, checkpoint_path=None, model_cfg=None, device="cuda", output_channel=None):
        super(SAM2VNet, self).__init__()

        if checkpoint_path:
            sam2 = build_sam2(model_cfg, checkpoint_path, device=device, apply_postprocessing=False)
        else:
            sam2 = build_sam2(model_cfg, device=device, apply_postprocessing=False)

        # 提取并保留Encoder的trunk部分
        self.encoder = sam2.image_encoder.trunk

        # print(f'encoder is {self.encoder}')

        # 修改PatchEmbed的Conv2d输入通道为1，同时修改stride为(2,2)防止图片分辨率过小
        self.encoder.patch_embed.proj = nn.Conv2d(
            1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        
        nn.init.kaiming_uniform_(
            self.encoder.patch_embed.proj.weight, a=math.sqrt(5))
        
        if self.encoder.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.encoder.patch_embed.proj.bias)

        # 定义Decoder的各个上采样模块
        self.upsample1 = Up(in_channels=768, out_channels=384)
        self.upsample2 = Up(in_channels=384, out_channels=192)
        self.upsample3 = Up(in_channels=192, out_channels=96)

        # 定义PatchExpand
        self.patchexpand = Final_PatchExpand2D(dim=96, dim_scale=2)

        # 定义输出卷积层
        self.outc = OutConv(
            in_channels=48, out_channels=output_channel)  # 最终输出
        
        # 定义VNet
        self.vnet = VNet(n_channels=1, n_classes=2, n_filters=16, has_dropout=True)

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

    def sam2_part(self, x):
        """SAM2部分，编码器使用SAM2，解码器使用普通的UNet的解码器先进行尝试"""
        encoder_outputs = self.encoder(x)  # 返回一个list，包含多个特征图

        # 检查模型的Encoder输出的内容
        x1 = encoder_outputs[0]  # (batch_size, 96, 56, 56)
        x2 = encoder_outputs[1]  # (batch_size, 192, 28, 28)
        x3 = encoder_outputs[2]  # (batch_size, 384, 14, 14)
        x4 = encoder_outputs[3]  #  (batch_size, 768, 7, 7)

        # 上采样过程
        x5 = self.upsample1(x4, x3)
        x6 = self.upsample2(x5, x2)
        x7 = self.upsample3(x6, x1)

        # x7输如PatchExpand和OutputConv
        x8 = self.patchexpand(x7.permute(0, 2, 3, 1))
        x8 = x8.permute(0, 3, 1, 2)
        sam2_output = self.outc(x8)

        return sam2_output

    def forward(self, x):
        print(f'input shape is {x.shape}')

        # 将3D数据切片为2D张量列表
        saminput_slices = self.slice_3d_to_2d(x)  # 调用类方法

        # 逐个将2D切片通过SAM2
        sam2_processed_slices = []
        for slice in saminput_slices:
            processed_slice = self.sam2_part(slice)
            sam2_processed_slices.append(processed_slice)

        # 将处理后的2D切片重新堆叠为3D张量
        sam_output = self.stack_2d_to_3d(sam2_processed_slices)

        # 直接使用x作为VNet的输入
        vnet_output = self.vnet(x)

        return sam_output,vnet_output


if __name__ == "__main__":
    # 检查CUDA是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU")

    # 创建Tiny SAM2模型进行尝试
    # sam2_checkpoint = "./sam2_configs/sam2_hiera_large.pt" # 对编码器冻结的情况下有较大的效果
    # sam2_checkpoint = "./sam2_configs/sam2.1_hiera_l.yaml"
    sam2_checkpoint = "./sam2_configs/sam2.1_hiera_l.yaml"
    model_cfg = "sam2.1_hiera_l.yaml"

    # 实例化SAM2UNet模型，不使用LoRA
    model = SAM2VNet(
        checkpoint_path=sam2_checkpoint,
        model_cfg=model_cfg,
        device=device,
        output_channel=2
    ).to(device)

    # 确认Encoder的参数是否被冻结
    # frozen_params = [p for p in sam2_unet.encoder.parameters() if not p.requires_grad]
    # print(f"Encoder冻结的参数数量: {len(frozen_params)}")

    # 测试前向传播
    dummy_input = torch.randn(1, 1, 112, 112, 80).to(device)  # 3D输入

    with torch.no_grad():
        try:
            sam_output,vnet_outut = model(dummy_input)
            print(f"sam输出张量形状: {sam_output.shape}")  # 应该为 (1, 2, 112, 112, 80)
            print(f"vnet输出张量形状: {vnet_outut.shape}")  # 应该为 (1, 2, 112, 112, 80)
        except Exception as e:
            print(f"前向传播时出错: {e}")
