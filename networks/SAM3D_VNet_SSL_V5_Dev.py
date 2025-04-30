"""
开发中
解决之前训练过程中遇到的一些问题
1. 在测试过程中分支有结果，但是平均下来结果却是0
2. 训练过程中权重更新是如何更新的
3. promptencoder相关的有一个tensor的shape是21而不是20，看看能不能解决
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

# 导入sam3D主要相关模块
from networks.sam_med3d.modeling.prompt_encoder3D import PromptEncoder3D
from networks.sam_med3d.modeling.mask_decoder3D import MaskDecoder3D

# 导入ImageEncoder其他模块
from networks.sam_med3d.modeling.image_encoder3D import Block3D
from networks.sam_med3d.modeling.image_encoder3D import PatchEmbed3D
from networks.sam_med3d.modeling.image_encoder3D import LayerNorm3d

# 导入粗分割VNet网络
from networks.VNet import VNet

class PromptGenerator_Encoder(nn.Module):
    """
    用于从粗分割中生成点Prompt
    """
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        normalization: Optional[str] = None,
        has_dropout: bool = False,
        pretrain_weight_path: Optional[str] = None,
        num_points_per_class: int = 5,
        threshold: float = 0.5,
        sample_mode: str = "random",  # 新增参数，支持 'random' 或 'topk'
        debug: bool = False,          # 是否输出调试信息
    ):
        super().__init__()
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0,1], got {threshold}")
        if num_points_per_class <= 0:
            raise ValueError(f"num_points_per_class must be >0, got {num_points_per_class}")
        assert sample_mode in ("random", "topk"), "sample_mode must be 'random' or 'topk'"

        # 使用VNet进行分割
        self.network = VNet(
            n_channels=n_channels,
            n_classes=n_classes,
            normalization=normalization,
            has_dropout=has_dropout
        )

        # 设置为调试模式
        self.debug = debug

        # 加载预训练权重（如有）
        if pretrain_weight_path:
            if self.debug:
                print(f'Loading pre-trained weights from {pretrain_weight_path}')
            self.load_model(self.network, pretrain_weight_path)

        self.num_points_per_class = num_points_per_class
        self.threshold = threshold
        self.sample_mode = sample_mode
        

    @staticmethod
    def load_model(model, model_path, device=None):
        """
        加载权重，自动处理单卡/多卡模型
        """
        state_dict = torch.load(model_path, map_location=device)
        if any(key.startswith('module.') for key in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        return model

    @staticmethod
    def _sample_points_from_mask(
        prob_map: torch.Tensor,
        label_value: int,
        num_points: int,
        threshold: float,
        mode: str = "random"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从单通道概率图 prob_map (shape = spatial_dims) 中，
        找到 prob > threshold 的所有坐标，采样num_points个点。
        支持随机采样或top-k采样。
        返回 coords (M, D) 和 labels (M,), label 全部为 label_value。
        """
        coords_all = (prob_map > threshold).nonzero(as_tuple=False)  # (K, D)
        K = coords_all.size(0)
        if K == 0:
            # 没有满足条件的点
            return (
                torch.empty(0, prob_map.dim(), device=prob_map.device, dtype=torch.long),
                torch.empty(0, dtype=torch.long, device=prob_map.device),
            )

        # 采样点数
        M = min(num_points, K)
        if mode == "random":
            idx = torch.randperm(K, device=prob_map.device)[:M]

        elif mode == "topk":
            # 取出所有满足条件点的概率
            values = prob_map[coords_all.unbind(dim=1)]
            _, topk_idx = torch.topk(values, M)
            idx = topk_idx
        else:
            raise ValueError(f"Unsupported sample mode: {mode}")

        sampled_coords = coords_all[idx]  # (M, D)
        sampled_labels = torch.full((M,), label_value, dtype=torch.long, device=prob_map.device)

        return sampled_coords, sampled_labels

    def _extract_point_prompts(
        self,
        logits: torch.Tensor,
        num_points_per_class: int,
        threshold: float,
        sample_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        核心提取函数
        logits: [B, C, *spatial]
        """
        B, C = logits.shape[:2]
        device = logits.device
        probs = F.softmax(logits, dim=1)

        batch_coords: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []

        for b in range(B):
            coords_per_sample: List[torch.Tensor] = []
            labels_per_sample: List[torch.Tensor] = []

            for c in range(1, C):
                pm = probs[b, c]
                fg_coords, fg_labels = self._sample_points_from_mask(
                    prob_map=pm,
                    label_value=c,
                    num_points=num_points_per_class,
                    threshold=threshold,
                    mode=sample_mode,
                )
                if self.debug:
                    print(f'[Class {c}] fg_coords shape: {fg_coords.shape}, fg_labels shape: {fg_labels.shape}')
                if fg_coords.numel() > 0:
                    coords_per_sample.append(fg_coords)
                    labels_per_sample.append(fg_labels)

            # 背景点采样（可按需优化为边界采样）
            pm_bg = probs[b, 0]
            bg_coords, bg_labels = self._sample_points_from_mask(
                prob_map=pm_bg,
                label_value=0,
                num_points=num_points_per_class,
                threshold=threshold,
                mode=sample_mode,
            )
            if self.debug:
                print(f'[Background] bg_coords shape: {bg_coords.shape}')
            if bg_coords.numel() > 0:
                coords_per_sample.append(bg_coords)
                labels_per_sample.append(bg_labels)

            # 若所有类别都未采到点，则加一个dummy点
            if not coords_per_sample:
                D = logits.dim() - 2
                coords_per_sample = [
                    torch.zeros(1, D, dtype=torch.long, device=device)
                ]
                labels_per_sample = [
                    torch.zeros(1, dtype=torch.long, device=device)
                ]

            # 拼接所有采样点
            coords_cat = torch.cat(coords_per_sample, dim=0)
            labels_cat = torch.cat(labels_per_sample, dim=0)
            # 添加到batch中
            batch_coords.append(coords_cat)
            batch_labels.append(labels_cat)

        # Pad到同样长度
        coords_padded = pad_sequence(batch_coords, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(batch_labels, batch_first=True, padding_value=-1)

        return coords_padded, labels_padded

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() < 3:
            raise ValueError(f"Input tensor must be at least 3D (B,C,spatial...), got {x.shape}")

        n_pts = self.num_points_per_class
        thr = self.threshold
        mode = self.sample_mode

        logits = self.network(x)  # [B, num_classes, *spatial]
        coords, labels = self._extract_point_prompts(logits, n_pts, thr, mode)
        return coords, labels

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

        # ------- Prompt Generator -------
        self.promptgenerator = PromptGenerator_Encoder(
            n_channels=self.in_channels,
            n_classes=self.n_classes,
            normalization=self.normalization,
            has_dropout=self.has_dropout,
            pretrain_weight_path=self.pretrain_weight_path,
            num_points_per_class=self.num_points_per_class,
            threshold=self.threshold,
            sample_mode=self.generatorways,
            debug=self.debug
        )

        # ------- Prompt Encoder -------
        self.promptencoder = PromptEncoder3D(
            embed_dim=self.embed_dim,                     # e.g., 768
            image_embedding_size=self.embedding_size,     # e.g., (14, 14, 10)
            input_image_size=self.image_size,                  # e.g., (112, 112, 80)
            mask_in_chans=self.mask_in_chans,               # e.g., 16
            activation=self.activation
        )

        # ------- Image Encoder -------
        self.samencoder = ImageEncoderViT3D(
            img_size=self.image_size,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=self.embed_dim,
            depth=self.encoder_depth,
            out_chans=self.out_chans
        )

        # ------- Mask Decoder -------
        self.maskdecoder = MaskDecoder3D(
            num_multimask_outputs=self.num_multimask_outputs,
            transformer_dim=self.embed_dim,
            iou_head_depth=self.iou_head_depth
        )

        # ------- VNet -------
        self.vnet = VNet(n_channels=self.in_channels,
                         n_classes=self.n_classes,
                         normalization=self.normalization,
                         has_dropout=self.has_dropout)

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        # Step 1: 从 Transformer 输出空间上采样回原始输入尺寸
        masks = F.interpolate(
            masks,
            size=input_size,
            mode="trilinear",
            align_corners=False,
        )

        # Step 2: 如果输入尺寸与原图尺寸不同，进一步对其
        if input_size != original_size:
            masks = F.interpolate(masks, size=original_size, mode="trilinear", align_corners=False)

        return masks

    def forward(self, x):
        # 0.VNet分支输出结果
        vnet_output = self.vnet(x)

        # 1. 图像主干编码
        after_encoder = self.samencoder(x)

        # 2. 获得prompt
        coords, labels = self.promptgenerator(x)

        # 3. prompt 编码
        sparse_embeddings, dense_embeddings = self.promptencoder(
            points=(coords, labels),
            boxes=None,
            masks=None
        )

        # 获得位置编码
        image_pe = self.promptencoder.get_dense_pe()

        # 4. 掩码解码
        after_maskencoder, iou_pred = self.maskdecoder(
            image_embeddings=after_encoder,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True
        )

        # 5. 插值还原尺寸
        sam_output = self.postprocess_masks(after_maskencoder, input_size=x.shape[-3:], original_size=x.shape[-3:])

        return vnet_output,sam_output


def networktest():
    # 检查CUDA是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU")

    # 实例化网络
    model = Network(in_channels=1,encoder_depth=8).to(device=device)

    # 冻结Network的ImageEncoder3D模块
    for param in model.samencoder.parameters():
        param.requires_grad = False

    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device=device)
    vnet_output, sam_output = model(input_tensor)
    print(f'VNet输出形状: {vnet_output.shape}')
    print(f'SAM输出形状: {sam_output.shape}')
    
    summary(model, input_size=(1, 1, 112, 112, 80), device=device)

    return 


if __name__ == "__main__":
    networktest()
