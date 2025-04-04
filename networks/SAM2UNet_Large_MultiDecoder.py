"""
编码器采用SAM2 Large的Encoder
不冻结Encoder/MaskDecoder
解码器使用UNet的Decoder与SAM的MaskDecoder
最后使用PatchExpand进行还原
在处理Unlabelled标签的时候使用MaskDecoder的作为伪标签进行训练
"""
import torch
import torch.nn as nn
from networks.sam2.build_sam import build_sam2
import math
import torch.nn.functional as F
from einops import rearrange
import copy
from typing import Tuple
from torchinfo import summary
from timm.layers import DropPath, to_2tuple, trunc_normal_

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

# 来自于Meta的LayerNorm操作
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# 来自于Meta的DropPath操作
class DropPath(nn.Module):
    # adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

# 来自于Meta的get_clones操作
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MaskDownSampler(nn.Module):
    """
    Progressively downsample a mask by total_stride, each time by stride.
    Note that LayerNorm is applied per *token*, like in ViT.

    With each downsample (by a factor stride**2), channel capacity increases by the same factor.
    In the end, we linearly project to embed_dim channels.
    """

    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=nn.GELU,
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))

    def forward(self, x):
        return self.encoder(x)


# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class CXBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Fuser(nn.Module):
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = get_clones(layer, num_layers)

        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # normally x: (N, C, H, W)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MemoryEncoder(nn.Module):
    def __init__(
        self,
        out_dim,
        mask_downsampler,
        fuser,
        position_encoding,
        in_dim=256,  # in_dim of pix_feats
    ):
        super().__init__()

        self.mask_downsampler = mask_downsampler

        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## Process masks
        # sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        ## Fuse pix_feats and downsampled masks
        # in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(masks.device)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": [pos]}

# 在Windows中计算注意力
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        构造函数

        参数:
            dim (int): 输入特征的通道数。
            window_size (tuple): 窗口大小，(Wh, Ww) 分别表示窗口的高度和宽度。
            num_heads (int): 注意力头的数量。
            qkv_bias (bool): 是否对 Q、K、V 的线性变换添加偏置。
            qk_scale (float): 缩放因子。如果为 None，则默认为 1/sqrt(每个头的通道数)。
            attn_drop (float): 注意力分数的丢弃率。
            proj_drop (float): 输出的丢弃率。
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # 窗口高度和宽度
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个头的通道数
        self.scale = qk_scale or head_dim ** -0.5  # 默认缩放因子

        # 相对位置编码表，形状为 (2 * Wh - 1) * (2 * Ww - 1) x nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  

        # 构建窗口内每两个像素之间的相对位置索引
        coords_h = torch.arange(self.window_size[0])  # 高度方向的坐标
        coords_w = torch.arange(self.window_size[1])  # 宽度方向的坐标
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 构建网格坐标
        coords_flatten = torch.flatten(coords, 1)  # 展平成 (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 计算相对位置，形状为 (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # 转置为 (Wh*Ww, Wh*Ww, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 相对位置的偏移处理，确保从0开始
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # 转换为一维索引
        relative_position_index = relative_coords.sum(-1)  # 索引值，形状为 (Wh*Ww, Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index)  # 保存为不可训练的常量

        # 定义Q、K、V投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 注意力分数的Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        # 输出投影层
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 初始化相对位置偏置表
        trunc_normal_(self.relative_position_bias_table, std=.02)
        # softmax 用于计算注意力分布
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        前向传播

        参数:
            x (torch.Tensor): 输入张量，形状为 (B_, N, C)，
                              其中 B_ 是窗口的总数，N 是每个窗口的Token数，C 是通道数。
            mask (torch.Tensor, 可选): 掩码张量，用于跨窗口的注意力。

        返回:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        B_, N, C = x.shape
        # 计算 Q、K、V
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 获取 Q、K、V

        # 缩放 Q
        q = q * self.scale
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1))

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  
        attn = attn + relative_position_bias.unsqueeze(0)

        # 如果存在掩码，添加到注意力分数中
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # Dropout 和计算最终输出
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def WindowsAttention_test():
    # 创建 WindowAttention 实例
    window_attention = WindowAttention(
        dim=128,           # 输入通道数
        window_size=(4, 4),  # 窗口大小
        num_heads=4,        # 注意力头数
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.1,
        proj_drop=0.1
    )

    # 模拟输入张量
    # 假设每个窗口的 Token 数为 4x4=16，通道数为 128，总窗口数为 8
    B_ = 8  # 总窗口数
    N = 16  # 每个窗口的 Token 数
    C = 128  # 通道数
    x = torch.randn(B_, N, C)

    # 测试无掩码的前向传播
    output = window_attention(x)
    print("Output shape without mask:", output.shape)

    # 测试带掩码的前向传播
    mask = torch.zeros(1, N, N)  # 一个简单的掩码示例
    output_with_mask = window_attention(x, mask)
    print("Output shape with mask:", output_with_mask.shape)


# 定义SAM2UNet模型
class SAM2UNet(nn.Module):
    """
    使用SAM2的Encoder和UNet的Decoder构建的多类分割网络。
    """
    def __init__(self, checkpoint_path=None, model_cfg="./configs/sam2.1/sam2.1_hiera_t.yaml", device="cuda", output_channel=None):
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

        # 提取并保留sam网络中的MemoryEncoder部分
        self.sam_memory_encoder = sam2.memory_encoder
        print(self.sam_memory_encoder)

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

def samnetwork_test():
    # 检查CUDA是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU")

    # 模型配置和检查点路径
    sam2_checkpoint = "./network/sam2/checkpoints/sam2.1_hiera_large.pt"  # 对编码器冻结的情况下有较大的效果
    model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"

    # 实例化SAM2UNet模型
    model = SAM2UNet(
        checkpoint_path=sam2_checkpoint,
        model_cfg=model_cfg,
        device=device,
        output_channel=9
    ).to(device)

    # 测试前向传播
    dummy_input = torch.randn(1, 1, 224, 224).to(device)  # 单通道输入
    with torch.no_grad():
        try:
            output = model(dummy_input).to(device)
            print(f"输出张量形状: {output.shape}")  # 应该为 (1, 9, 224, 224)
        except Exception as e:
            print(f"前向传播时出错: {e}")


# 测试前向传播
if __name__ == "__main__":
    WindowsAttention_test()