"""
使用粗分割形成Prompt
利用SAM3D的模块实现半监督网络
暂时使用伪标签的做法
暂时不使用预加载的权重
"""

import sys
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange
from torchinfo import summary
from typing import Tuple, Optional,Type,Union

# 导入sam3D主要相关模块
# from networks.sam_med3d.modeling.image_encoder3D import ImageEncoderViT3D
from networks.sam_med3d.modeling.prompt_encoder3D import PromptEncoder3D

# 导入ImageEncoder其他模块
from networks.sam_med3d.modeling.image_encoder3D import Block3D
from networks.sam_med3d.modeling.image_encoder3D import PatchEmbed3D
from networks.sam_med3d.modeling.image_encoder3D import LayerNorm3d

# 导入VNet网络
from networks.VNet import VNet

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

class PromptGenerator_Encoder(nn.Module):
    """
    生成point的prompt
    这里输入的张量是BCHWD
    输出的点坐标对应的是HWD，别弄错了
    """
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        normalization: Optional[str] = None,
        has_dropout: bool = False,
        pretrain_weight_path: Optional[str] = None,
        num_points_per_class: int = 5,
        threshold: float = 0.5
    ):
        super().__init__()
        self.network = VNet(
            n_channels=n_channels,
            n_classes=n_classes,
            normalization=normalization,
            has_dropout=has_dropout
        )
        self.pretrain_path = pretrain_weight_path
        self.num_points_per_class = num_points_per_class
        self.threshold = threshold

    def _extract_point_prompts(
        self,
        input_tensor: torch.Tensor,
        num_points_per_class: int,
        threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract point prompts from segmentation logits."""

        device = input_tensor.device
        batch_size, num_classes = input_tensor.shape[0], input_tensor.shape[1]
        prob_tensor = F.softmax(input_tensor, dim=1)
        
        # Determine spatial dimension (2D or 3D)
        ndim = input_tensor.ndim - 2
        coord_dim = ndim  # 2 for (y,x), 3 for (z,y,x)

        coords_list, labels_list = [], []
        for b in range(batch_size):
            batch_coords, batch_labels = [], []
            has_sampled_background = False

            for c in range(1, num_classes):  # Skip background (c=0)
                prob_map = prob_tensor[b, c]
                fg_indices = torch.where(prob_map > threshold)

                if len(fg_indices[0]) > 0:
                    # Sample foreground points
                    fg_sample_idx = torch.randperm(len(fg_indices[0]))[:num_points_per_class]
                    fg_coords = torch.stack([fg_indices[i][fg_sample_idx] for i in range(coord_dim)], dim=-1)
                    fg_labels = torch.ones(len(fg_sample_idx), dtype=torch.long, device=device)
                    batch_coords.append(fg_coords)
                    batch_labels.append(fg_labels)

                # Sample background points once per batch
                if not has_sampled_background:
                    bg_prob_map = prob_tensor[b, 0]
                    bg_indices = torch.where(bg_prob_map > threshold)
                    if len(bg_indices[0]) > 0:
                        bg_sample_idx = torch.randperm(len(bg_indices[0]))[:num_points_per_class]
                        bg_coords = torch.stack([bg_indices[i][bg_sample_idx] for i in range(coord_dim)], dim=-1)
                        bg_labels = torch.zeros(len(bg_sample_idx), dtype=torch.long, device=device)
                        batch_coords.append(bg_coords)
                        batch_labels.append(bg_labels)
                        has_sampled_background = True

            # Handle empty batches
            if batch_coords:
                batch_coords = torch.cat(batch_coords, dim=0)
                batch_labels = torch.cat(batch_labels, dim=0)
            else:
                batch_coords = torch.zeros(0, coord_dim, dtype=torch.float32, device=device)
                batch_labels = torch.zeros(0, dtype=torch.long, device=device)

            coords_list.append(batch_coords)
            labels_list.append(batch_labels)

        # Pad to max points in batch
        max_points = max([c.shape[0] for c in coords_list]) if coords_list else 0
        if max_points > 0:
            coords = torch.zeros(batch_size, max_points, coord_dim, dtype=torch.float32, device=device)
            labels = torch.zeros(batch_size, max_points, dtype=torch.long, device=device)
            for b in range(batch_size):
                num_points = coords_list[b].shape[0]
                if num_points > 0:
                    coords[b, :num_points] = coords_list[b]
                    labels[b, :num_points] = labels_list[b]
                labels[b, num_points:] = -1  # Pad with -1
        else:
            coords = torch.zeros(batch_size, 1, coord_dim, dtype=torch.float32, device=device)
            labels = torch.full((batch_size, 1), -1, dtype=torch.long, device=device)

        return coords, labels

    def forward(
        self,
        x: torch.Tensor,
        num_points_per_class: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_points = num_points_per_class or self.num_points_per_class
        threshold = threshold or self.threshold

        input_tensor = self.network(x)
        coords, labels = self._extract_point_prompts(input_tensor, num_points, threshold)

        return coords, labels

def PromptGenerator_Encoder_Test():
    """
    测试promptencoder
    """
    device = torch.device("cuda")
    inputtensor = torch.randn(1,1,112,112,80).to(device=device)
    print(f'input shape is {inputtensor.shape}')
    promptencoder = PromptGenerator_Encoder(n_channels=1, n_classes=2, normalization="batchnorm",has_dropout=True,
                                            pretrain_weight_path=None,num_points_per_class=10,threshold=0.5).to(device=device)
    coords,labels = promptencoder(inputtensor)

    return coords, labels

def PromptEncoder_Test():
    """
    用于单独测试PromptEncoder3D
    验证张量的输入与输出顺序
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化编码器
    encoder = PromptEncoder3D(
        embed_dim=256,
        image_embedding_size=(5, 7, 7),  # 假设ImageEncoder下采样后尺寸
        input_image_size=(80, 112, 112),  # 原始图像尺寸 (D,H,W)
        mask_in_chans=16
    ).to(device)
    
    # 模拟输入
    coords = torch.randint(0, [80, 112, 112], (1, 10, 3)).float().to(device)  # [B,N,(D,H,W)]
    labels = torch.randint(0, 2, (1, 10)).to(device)  # [B,N]
    masks = torch.rand(1, 1, 80, 112, 112).to(device)  # [B,1,D,H,W]
    
    # 前向传播
    sparse_emb, dense_emb = encoder(
        points=(coords, labels),
        boxes=None,
        masks=masks
    )
    
    # 验证输出形状
    print(f"Sparse embeddings shape: {sparse_emb.shape} (应为 [1, 10, 256])")
    print(f"Dense embeddings shape: {dense_emb.shape} (应为 [1, 256, 5, 7, 7])")

    return

def PromptEncoder_Test_Full():
    """
    测试从Prompt生成到编码的完整流程
    流程：PromptGenerator_Encoder -> PromptEncoder3D
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 生成模拟输入图像 (B=1, C=1, D=80, H=112, W=112)
    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device)
    print(f"\n[Input] Image shape: {input_tensor.shape}")

    # 2. 初始化Prompt生成器 (输出20个点)
    prompt_generator = PromptGenerator_Encoder(
        n_channels=1,
        n_classes=2,
        normalization="batchnorm",
        num_points_per_class=10,  # 每类10点 -> 共20点 (10前景+10背景)
        threshold=0.5
    ).to(device)
    
    # 3. 生成点提示
    coords, labels = prompt_generator(input_tensor)
    print(f"\n[PromptGenerator Output]")
    print(f"Coordinates shape: {coords.shape} (B, N_points, 3)")
    print(f"Labels shape: {labels.shape} (B, N_points)")
    print(f"Sample coordinates (Z,Y,X):\n{coords[0, :5]}")  # 打印前5个点
    print(f"Sample labels:\n{labels[0, :5]}")

    # 4. 初始化Prompt编码器
    prompt_encoder = PromptEncoder3D(
        embed_dim=256,
        image_embedding_size=(7, 7, 5),  # 需与ImageEncoder输出对齐 (D',H',W')
        input_image_size=(80, 112, 112),  # 原始图像尺寸 (D,H,W)
        mask_in_chans=16,
        activation=nn.GELU
    ).to(device)

    # 5. 编码点提示 (假设无mask输入)
    sparse_embeddings, dense_embeddings = prompt_encoder(
        points=(coords, labels),
        boxes=None,
        masks=None
    )
    
    # 6. 打印编码结果
    print(f"\n[PromptEncoder Output]")
    print(f"Sparse embeddings shape: {sparse_embeddings.shape} (B, N_points, embed_dim)")
    print(f"Dense embeddings shape: {dense_embeddings.shape} (B, embed_dim, D', H', W')")
    print(f"Sparse embeddings mean: {sparse_embeddings.mean().item():.4f}")
    print(f"Dense embeddings mean: {dense_embeddings.mean().item():.4f}")

    # 7. 验证与ImageEncoder输出的兼容性
    image_encoder_output = torch.randn(1, 256, 7, 7, 5).to(device)  # 模拟ImageEncoder输出
    assert dense_embeddings.shape == image_encoder_output.shape, \
        f"Shape mismatch: {dense_embeddings.shape} vs {image_encoder_output.shape}"
    print("\n[Validation] Dense embeddings与ImageEncoder输出形状兼容")

    return {
        "input_image": input_tensor,
        "coordinates": coords,
        "labels": labels,
        "sparse_embeddings": sparse_embeddings,
        "dense_embeddings": dense_embeddings
    }

class ImageEncoderViT3D(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = (256, 256, 256),  # 支持动态尺寸
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        
        super().__init__()
        # 处理img_size为三元组（兼容旧版int输入）
        if isinstance(img_size, int):
            img_size = (img_size, img_size, img_size)
        self.img_size = img_size
        self.patch_size = patch_size

        # Patch Embedding
        self.patch_embed = PatchEmbed3D(
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # 动态位置编码
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    img_size[0] // patch_size,  # D
                    img_size[1] // patch_size,  # H
                    img_size[2] // patch_size,  # W
                    embed_dim
                )
            )

        # Transformer Blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block3D(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(
                    img_size[0] // patch_size,
                    img_size[1] // patch_size,
                    img_size[2] // patch_size,
                ),
            )
            self.blocks.append(block)

        # Neck Network
        self.neck = nn.Sequential(
            nn.Conv3d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm3d(out_chans),
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm3d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 检查输入尺寸合法性
        assert x.ndim == 5, f"Input must be 5D [B,C,D,H,W], got {x.shape}"
        assert all([s % self.patch_size == 0 for s in x.shape[-3:]]), \
            f"Input spatial size {x.shape[-3:]} must be divisible by patch_size {self.patch_size}"

        # Patch Embedding
        x = self.patch_embed(x)  # [B, D', H', W', C]
        
        # 位置编码
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Transformer Blocks
        for blk in self.blocks:
            x = blk(x)

        # 转换维度并输出 [B, C, D', H', W']
        x = self.neck(x.permute(0, 4, 1, 2, 3))

        return x

def test_imageencoder_vit3d():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 非正方形输入 (D=80, H=112, W=112)
    input_tensor = torch.randn(1, 1, 80, 112, 112).to(device)
    
    # 初始化编码器 (patch_size需能整除输入尺寸)
    encoder = ImageEncoderViT3D(
        img_size=(80, 112, 112),  # 指定非正方形尺寸
        patch_size=16,            # 80/16=5, 112/16=7
        in_chans=1,
        embed_dim=768,
    ).to(device)
    
    # 前向传播
    output = encoder(input_tensor)
    print(f"Input shape: {input_tensor.shape}")   # [1,1,80,112,112]
    print(f"Output shape: {output.shape}")        # [1,256,5,7,7]

class Network(nn.Module):
    def __init__(self, in_channel=1, 
                 output_channel=2,
                 image_size=(80,112,112),
                 patchsize=16,
                 embed_dim=768,
                 depth=12,
                 numheads=12
    ):
        super(Network, self).__init__()

        # 定义编码器ImageEncoderVit3D
        self.samencoder = ImageEncoderViT3D(img_size=image_size,
                                            patch_size=patchsize,
                                            in_chans=in_channel,
                                            embed_dim=embed_dim,
                                            depth=depth,
                                            num_heads=numheads
                                            )

    def forward(self, x):
        # 通过encoder
        print(f'input shape is {x.shape}')
        # 调整输入顺序
        x = x.permute(0,1,4,2,3)
        print(f'after permute x shape is {x.shape}')

        after_encoder = self.samencoder(x)

        print(f'after encoder shape is {after_encoder.shape}')
        after_encoder = after_encoder.permute(0,1,3,4,2)
        print(f'after permute shape is {after_encoder.shape}')

        return

def networktest():
    # 检查CUDA是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU")

    # 实例化网络
    model = Network(in_channel=1,
                     output_channel=2,
                     image_size=(80,112,112),
                     patchsize=16,
                     embed_dim=768,
                     depth=12,
                     numheads=12
                    ).to(device=device)

    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device=device)
    model(input_tensor)
    return

if __name__ == "__main__":
    # PromptGenerator_Encoder_Test()
    # test_imageencoder_vit3d()
    # networktest()
    PromptEncoder_Test()