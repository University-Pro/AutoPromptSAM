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
from typing import Tuple, Optional, Type, Union

# 导入sam3D主要相关模块
# from networks.sam_med3d.modeling.image_encoder3D import ImageEncoderViT3D
from networks.sam_med3d.modeling.prompt_encoder3D import PromptEncoder3D
from networks.sam_med3d.modeling.mask_decoder3D import MaskDecoder3D

# 导入ImageEncoder其他模块
from networks.sam_med3d.modeling.image_encoder3D import Block3D
from networks.sam_med3d.modeling.image_encoder3D import PatchEmbed3D
from networks.sam_med3d.modeling.image_encoder3D import LayerNorm3d

# 导入VNet网络
from networks.VNet import VNet


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
                    fg_sample_idx = torch.randperm(len(fg_indices[0]))[
                        :num_points_per_class]
                    fg_coords = torch.stack(
                        [fg_indices[i][fg_sample_idx] for i in range(coord_dim)], dim=-1)
                    fg_labels = torch.ones(
                        len(fg_sample_idx), dtype=torch.long, device=device)
                    batch_coords.append(fg_coords)
                    batch_labels.append(fg_labels)

                # Sample background points once per batch
                if not has_sampled_background:
                    bg_prob_map = prob_tensor[b, 0]
                    bg_indices = torch.where(bg_prob_map > threshold)
                    if len(bg_indices[0]) > 0:
                        bg_sample_idx = torch.randperm(len(bg_indices[0]))[
                            :num_points_per_class]
                        bg_coords = torch.stack(
                            [bg_indices[i][bg_sample_idx] for i in range(coord_dim)], dim=-1)
                        bg_labels = torch.zeros(
                            len(bg_sample_idx), dtype=torch.long, device=device)
                        batch_coords.append(bg_coords)
                        batch_labels.append(bg_labels)
                        has_sampled_background = True

            # Handle empty batches
            if batch_coords:
                batch_coords = torch.cat(batch_coords, dim=0)
                batch_labels = torch.cat(batch_labels, dim=0)
            else:
                batch_coords = torch.zeros(
                    0, coord_dim, dtype=torch.float32, device=device)
                batch_labels = torch.zeros(0, dtype=torch.long, device=device)

            coords_list.append(batch_coords)
            labels_list.append(batch_labels)

        # Pad to max points in batch
        max_points = max([c.shape[0]
                         for c in coords_list]) if coords_list else 0
        if max_points > 0:
            coords = torch.zeros(batch_size, max_points,
                                 coord_dim, dtype=torch.float32, device=device)
            labels = torch.zeros(batch_size, max_points,
                                 dtype=torch.long, device=device)
            for b in range(batch_size):
                num_points = coords_list[b].shape[0]
                if num_points > 0:
                    coords[b, :num_points] = coords_list[b]
                    labels[b, :num_points] = labels_list[b]
                labels[b, num_points:] = -1  # Pad with -1
        else:
            coords = torch.zeros(batch_size, 1, coord_dim,
                                 dtype=torch.float32, device=device)
            labels = torch.full((batch_size, 1), -1,
                                dtype=torch.long, device=device)

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
        coords, labels = self._extract_point_prompts(
            input_tensor, num_points, threshold)
        # print(f'coords shape is {coords.shape}')
        # print(f'labels shape is {labels.shape}')

        return coords, labels


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
    print(f"Sample coordinates (XYZ):\n{coords[0, :5]}")  # 打印前5个点
    print(f"Sample labels:\n{labels[0, :5]}")

    # 4. 初始化Prompt编码器
    prompt_encoder = PromptEncoder3D(
        embed_dim=256,
        image_embedding_size=(14, 14, 10),  # 需与ImageEncoder输出对齐 (D',H',W')
        input_image_size=(112, 112, 80),
        mask_in_chans=16,
        activation=nn.GELU
    ).to(device)

    # 5. 编码点提示
    sparse_embeddings, dense_embeddings = prompt_encoder(
        points=(coords, labels),
        boxes=None,
        masks=None
    )

    # 6. 打印编码结果
    print(f"\n[PromptEncoder Output]")
    print(
        f"Sparse embeddings shape: {sparse_embeddings.shape} (B, N_points, embed_dim)")
    print(
        f"Dense embeddings shape: {dense_embeddings.shape} (B, embed_dim, HWD')")
    print(f"Sparse embeddings mean: {sparse_embeddings.mean().item():.4f}")
    print(f"Dense embeddings mean: {dense_embeddings.mean().item():.4f}")

    return


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
            nn.Conv3d(out_chans, out_chans, kernel_size=3,
                      padding=1, bias=False),
            LayerNorm3d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 检查输入尺寸合法性
        assert x.ndim == 5, f"Input must be 5D [B,C,D,H,W], got {x.shape}"
        assert all([s % self.patch_size == 0 for s in x.shape[-3:]]), \
            f"Input spatial size {x.shape[-3:]} must be divisible by patch_size {self.patch_size}"

        # Patch Embedding
        x = self.patch_embed(x)
        # print(f'after patchembed shape is {x.shape}')

        # 位置编码
        if self.pos_embed is not None:
            x = x + self.pos_embed
        # print(f'after pos_embed shape is {x.shape}')
        # Transformer Blocks
        for blk in self.blocks:
            x = blk(x)
            # print(f'after block shape is {x.shape}')

        # 转换维度并输出 [B, C, D', H', W']
        x = x.permute(0, 4, 1, 2, 3)
        # print(f'after permute shape is {x.shape}')
        x = self.neck(x)

        return x


def test_imageencoder_vit3d():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 非正方形输入 (D=80, H=112, W=112)
    # input_tensor = torch.randn(1, 1, 80, 112, 112).to(device)
    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device)  # BCHWD
    print(f"Input shape: {input_tensor.shape}")

    # 初始化编码器 (patch_size需能整除输入尺寸)
    encoder = ImageEncoderViT3D(
        # img_size=(80, 112, 112),  # 指定非正方形尺寸
        img_size=(112, 112, 80),
        patch_size=8,            # 80/16=5, 112/16=7
        in_chans=1,
        embed_dim=768,
        depth=12
    ).to(device)

    # 前向传播
    output = encoder(input_tensor)

    print(f"Output shape: {output.shape}")        # [1,256,5,7,7]


def test_maskdecoder():
    """
    用于测试导入的maskencoder
    """
    # 创建 MaskDecoder3D 类的实例
    transformer_dim = 128
    num_multimask_outputs = 3
    model = MaskDecoder3D(transformer_dim=transformer_dim,
                          num_multimask_outputs=num_multimask_outputs)

    # 模拟输入
    batch_size = 2
    image_embeddings = torch.randn(
        batch_size, transformer_dim, 32, 32, 32)  # 图像嵌入（假设大小）
    image_pe = torch.randn(batch_size, transformer_dim, 32, 32, 32)  # 位置编码
    sparse_prompt_embeddings = torch.randn(
        batch_size, 5, transformer_dim)  # 稀疏提示嵌入（例如 5 个点或框）
    dense_prompt_embeddings = torch.randn(
        batch_size, transformer_dim, 32, 32, 32)  # 密集提示嵌入（掩码）
    multimask_output = True  # 是否输出多个掩码

    # 前向传播
    masks, iou_pred = model(image_embeddings, image_pe,
                            sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output)

    # 输出形状
    print(f"Masks Shape: {masks.shape}")  # 输出的掩码形状
    print(f"IOU Prediction Shape: {iou_pred.shape}")  # 输出的 IOU 预测形状

    return


class Network(nn.Module):
    def __init__(self, pretrain_weight_path=None, num_points_per_class=10, threshold=0.5):
        super(Network, self).__init__()

        # 定义PromptGenerator
        self.promptgenerator = PromptGenerator_Encoder(n_channels=1, n_classes=2, normalization="batchnorm",
                                                       has_dropout=True, pretrain_weight_path=None, num_points_per_class=10, threshold=0.5)

        # 定义PromptEncoder
        self.promptencoder = PromptEncoder3D(embed_dim=768, image_embedding_size=(14, 14, 10),
                                             input_image_size=(112, 112, 80), mask_in_chans=16, activation=nn.GELU)
        # 定义编码器ImageEncoderVit3D
        self.samencoder = ImageEncoderViT3D(img_size=(112,112,80),patch_size=8,in_chans=1,
                                            embed_dim=768,depth=12,out_chans=768)

        # 定义MaskDecoder
        self.maskdecoder = MaskDecoder3D(
            num_multimask_outputs=2, transformer_dim=768, iou_head_depth=3)  # 修改 transformer_dim 为 768

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        # Step 1: 将 mask 从 Transformer 输出的空间，先上采样到网络输入大小（粗采样）
        # 原图为 112x112x80，编码器输出为 14x14x10，所以实际可以插值回原图大小
        masks = F.interpolate(
            masks,
            size=input_size,  # 也可以写成 (112, 112, 80)
            mode="trilinear",
            align_corners=False,
        )

        # Step 2: 如果输入尺寸与原图尺寸不同，这一步用于修整回原图尺寸
        if input_size != original_size:
            masks = F.interpolate(masks, size=original_size, mode="trilinear", align_corners=False)

        return masks

    def forward(self, x):
        # 通过encoder
        # print(f'input shape is {x.shape}')
        # 通过imageencoder
        after_encoder = self.samencoder(x)
        # print(f'after encoder shape is {after_encoder.shape}')

        # 获得粗分割prompt
        coords, labels = self.promptgenerator(x)
        # print(f'coords shape is {coords.shape}')
        # print(f'labels shape is {labels.shape}')

        # prompt通过promptencoder
        sparse_embeddings, dense_embeddings = self.promptencoder(
            points=(coords, labels),boxes=None,masks=None)
        # print(f'spares embeddings shape is {sparse_embeddings.shape}')
        # print(f'dense embeddings shape is {dense_embeddings.shape}')

        # 获得pe
        image_pe = self.promptencoder.get_dense_pe()
        # print(f'image_pe shape is {image_pe.shape}')

        # 设置maskencoder
        after_maskencoder,iou_pred = self.maskdecoder(image_embeddings=after_encoder,
                                                      image_pe=image_pe,
                                                      sparse_prompt_embeddings=sparse_embeddings,
                                                      dense_prompt_embeddings=dense_embeddings,
                                                      multimask_output=True)

        # print(f'mask shape is {after_maskencoder.shape}')
        # print(f'iou_pred shape is {iou_pred.shape}')

        # 把低分辨率的mask扩展到高分辨率
        mask = self.postprocess_masks(after_maskencoder, input_size=x.shape[-3:], original_size=x.shape[-3:])
        # print(f'mask shape is {mask.shape}')

        return mask


def networktest():
    # 检查CUDA是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU")

    # 实例化网络
    model = Network().to(device=device)

    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device=device)
    model(input_tensor)

    summary(model=model, input_size=(1, 1, 112, 112, 80))

    return


if __name__ == "__main__":
    # test_imageencoder_vit3d()
    # PromptEncoder_Test_Full()
    networktest()
