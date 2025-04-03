"""
编码器采用SAM2的Encoder
解码器使用UNet的Decoder
最后使用PatchExpand进行还原
进行全部训练
采用的是SAM2-Large的结构
"""
import sys
sys.path.append('/samba/network-storage/ssd/home/pi/sam2-test')  # 设置运行目录

import torch
import torch.nn as nn
from network.sam2.build_sam import build_sam2
import math
import torch.nn.functional as F
from einops import rearrange
from torchinfo import summary

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
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x

# 双层卷积
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# 上采样部分
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels=out_channels * 2, out_channels=out_channels)

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

class SAM2UNet(nn.Module):
    """
    使用SAM2的Encoder和UNet的Decoder构建的多类分割网络。
    """
    def __init__(self, checkpoint_path=None, model_cfg="./configs/sam2.1/sam2.1_hiera_t.yaml", device="cuda", output_channel = None):
        super(SAM2UNet, self).__init__()

        # 构建完整的SAM2模型
        if checkpoint_path:
            sam2 = build_sam2(model_cfg, checkpoint_path, device=device, apply_postprocessing=False)
        else:
            sam2 = build_sam2(model_cfg, device=device, apply_postprocessing=False)

        # 提取并保留Encoder的trunk部分
        self.encoder = sam2.image_encoder.trunk

        # 修改PatchEmbed的Conv2d输入通道为1，以接受单通道图像
        self.encoder.patch_embed.proj = nn.Conv2d(1, 144, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        nn.init.kaiming_uniform_(self.encoder.patch_embed.proj.weight, a=math.sqrt(5))
        if self.encoder.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.encoder.patch_embed.proj.bias)

        # 定义Decoder的各个上采样模块
        self.upsample1 = Up(in_channels=1152, out_channels=576)
        self.upsample2 = Up(in_channels=576, out_channels=288)
        self.upsample3 = Up(in_channels=288, out_channels=144)

        # 定义PatchExpand
        self.patchexpand = Final_PatchExpand2D(dim=144, dim_scale=4)

        # 定义输出卷积层
        self.outc = OutConv(in_channels=36, out_channels=output_channel)  # 最终输出


    def forward(self, x):
        # Encoder前向传播，获取多个尺度的特征图
        encoder_outputs = self.encoder(x)  # 返回一个list，包含多个特征图

        # 检查模型的Encoder输出的内容
        x1 = encoder_outputs[0]  # (1,144,56,56)
        x2 = encoder_outputs[1]  # (1,288,28,28)
        x3 = encoder_outputs[2]  # (1,576,14,14)
        x4 = encoder_outputs[3]  # (1,1152,7,7)

        # 上采样过程
        x5 = self.upsample1(x4, x3)
        x6 = self.upsample2(x5, x2)
        x7 = self.upsample3(x6, x1)

        # x7输如PatchExpand和OutputConv
        x8 = self.patchexpand(x7.permute(0,2,3,1))
        x8 = x8.permute(0,3,1,2)
        logits = self.outc(x8)

        # 在调试过程中使用模拟输出
        # logits = torch.randn(1,9,224,224)

        return logits

if __name__ == "__main__":
    # 检查CUDA是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU")

    # 模型配置和检查点路径
    sam2_checkpoint = "./sam2_configs/sam2_hiera_large.pt" # 对编码器冻结的情况下有较大的效果
    # model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"
    model_cfg = "sam2_hiera_l.yaml"

    # 实例化SAM2UNet模型，不使用LoRA
    sam2_unet = SAM2UNet(
        checkpoint_path=sam2_checkpoint,
        model_cfg=model_cfg,
        device=device,
        output_channel=9
    ).to(device)

    summary(model=sam2_unet,input_size=(1,1,224,224))

    # 测试前向传播
    dummy_input = torch.randn(1, 1, 224, 224).to(device)  # 单通道输入
    print(f'input shape is {dummy_input.shape}')
    with torch.no_grad():
        try:
            output = sam2_unet(dummy_input)
            print(f"输出张量形状: {output.shape}")  # 应该为 (1, 9, 224, 224)
        except Exception as e:
            print(f"前向传播时出错: {e}")
