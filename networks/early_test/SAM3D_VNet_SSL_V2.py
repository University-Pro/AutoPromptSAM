"""


使用粗分割形成Prompt
利用SAM3D的模块实现网络
相比于第一版本优化了一些代码
同时加入了粗分割的VNet的权重文件
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

# 导入LA数据集做测试用
from dataloader.DataLoader_LA import LAHeart
from utils.ImageAugment import RandomRotFlip_LA as RandomRotFlip
from utils.ImageAugment import RandomCrop_LA as RandomCrop
from utils.ImageAugment import ToTensor_LA as ToTensor
from torchvision import transforms

class PromptGenerator_Encoder(nn.Module):
    """
    从分割logits中提取点提示（point prompts）。
    输入 logits: Tensor[B, num_classes, *spatial_dims]
    输出 coords: Tensor[B, N_max, D], labels: Tensor[B, N_max]
        - D = len(spatial_dims)
        - N_max = batch中所有样本的最大点数
        - padding的label为-1
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
        """
        x: 输入影像 [B, in_chans, *spatial]
        返回 (coords, labels)
        """
        if x.dim() < 3:
            raise ValueError(f"Input tensor must be at least 3D (B,C,spatial...), got {x.shape}")

        n_pts = self.num_points_per_class
        thr = self.threshold
        mode = self.sample_mode

        logits = self.network(x)  # [B, num_classes, *spatial]
        coords, labels = self._extract_point_prompts(logits, n_pts, thr, mode)
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
        has_dropout=True,
        threshold=0.5,
        pretrain_weight_path="./result/VNet/LA/Pth/best.pth",
        num_points_per_class=10,  # 每类10点 -> 共20点 (10前景+10背景)
        sample_mode="random",     # 随机采样
        debug=True               # 打印调试信息
    ).to(device)

    # 3. 生成点提示
    coords, labels = prompt_generator(input_tensor)
    print(f"Coordinates shape: {coords.shape} (B, N_points, 3)")
    print(f"Labels shape: {labels.shape} (B, N_points)")
    print(f"Sample coordinates (XYZ):\n{coords[0, :20]}")  # 打印前5个点
    print(f"Sample labels:\n{labels[0, :20]}")
    print("All sampled labels:", labels[0])

    # # 4. 初始化Prompt编码器
    # prompt_encoder = PromptEncoder3D(
    #     embed_dim=256,
    #     image_embedding_size=(14, 14, 10),  # 需与ImageEncoder输出对齐 (D',H',W')
    #     input_image_size=(112, 112, 80),
    #     mask_in_chans=16,
    #     activation=nn.GELU
    # ).to(device)

    # # 5. 编码点提示
    # sparse_embeddings, dense_embeddings = prompt_encoder(
    #     points=(coords, labels),
    #     boxes=None,
    #     masks=None
    # )

    # # 6. 打印编码结果
    # print(f"\n[PromptEncoder Output]")
    # print(
    #     f"Sparse embeddings shape: {sparse_embeddings.shape} (B, N_points, embed_dim)")
    # print(
    #     f"Dense embeddings shape: {dense_embeddings.shape} (B, embed_dim, HWD')")
    # print(f"Sparse embeddings mean: {sparse_embeddings.mean().item():.4f}")
    # print(f"Dense embeddings mean: {dense_embeddings.mean().item():.4f}")

    return

def PromptEncoder_Test_Pic():
    """
    带有数据集可视化的图像测试
    流程：
      1) 从 LAHeart dataset 读取样本
      2) 用 PromptGenerator_Encoder 生成 sparse prompts (coords, labels)
      3) 在中间切片上可视化原图 & 采样点
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    patch_size = (112, 112, 80)
    train_transforms = transforms.Compose([
        RandomRotFlip(),
        RandomCrop(patch_size),
        ToTensor(),
    ])

    # 1) 构建数据集
    db_train = LAHeart(
        base_dir="./datasets/LA",
        split="train",
        transform=train_transforms,
        num=16
    )

    # 2) 初始化 Prompt 生成器
    prompt_generator = PromptGenerator_Encoder(
        n_channels=1,
        n_classes=2,
        normalization="batchnorm",
        num_points_per_class=50,  # 每类 50 个点
        threshold=0.5
    ).to(device)

    # 3) 遍历前 5 个样本
    for idx in range(min(5, len(db_train))):
        sample = db_train[idx]
        # image: [C, H, W, D] -> unsqueeze batch -> [1, C, H, W, D]
        image = sample["image"].unsqueeze(0).to(device)
        # label: [H, W, D] -> unsqueeze batch -> [1, H, W, D]
        label = sample["label"].unsqueeze(0).to(device)
        # 如果需要通道维，可加一行： label = label.unsqueeze(1)

        B, C, H, W, D = image.shape
        print(f"Sample {idx}: image {image.shape}, label {label.shape}")

        # 4) 生成 sparse prompts
        coords, point_labels = prompt_generator(image)
        # coords: [B, N_pts, 3], point_labels: [B, N_pts]
        coords_np = coords[0].cpu().numpy().astype(int)
        labels_np = point_labels[0].cpu().numpy().astype(int)

        # 5) 在深度中间切片上可视化
        img_np = image.squeeze().cpu().numpy()  # [H, W, D] (C=1 已被 squeeze)
        mid_z = D // 2
        slice_img = img_np[:, :, mid_z]         # XY 平面

        plt.figure(figsize=(5, 5))
        plt.imshow(slice_img, cmap="gray")
        plt.title(f"Sample {idx} - Z slice {mid_z}")

        # 只绘制落在此切片上的点
        mask = coords_np[:, 2] == mid_z
        pts = coords_np[mask]
        lbls = labels_np[mask]

        for (x, y, z), lbl in zip(pts, lbls):
            color = "r" if lbl == 1 else "b"
            plt.scatter(x, y, c=color, s=30, marker="x")

        plt.axis("off")
        plt.savefig(f"sample_{idx}_slice_{mid_z}.png",
                    bbox_inches="tight", dpi=300)
        plt.close()

        print(f"  -> sampled points total: {coords_np.shape[0]}, on mid-slice: {pts.shape[0]}\n")

    print("PromptEncoder_Test_Pic 完成。")

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
    def __init__(
            self,
            in_channels: int = 1,
            image_size: Tuple[int, int, int] = (112, 112, 80),
            patch_size: int = 8,
            embed_dim: int = 768,
            depth: int = 12,
            out_chans: int = 768,
            num_classes: int = 2,
            normalization: str = "batchnorm",
            has_dropout: bool = True,
            pretrain_weight_path: str = None,
            num_points_per_class: int = 10,
            threshold: float = 0.5,
            mask_in_chans: int = 16,
            activation=nn.GELU,
            num_multimask_outputs: int = 2,
            iou_head_depth: int = 3
    ):
        super(Network, self).__init__()

        # ------- 计算一些中间维度 -------
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_size = tuple(s // patch_size for s in image_size)  # e.g., 112 -> 14

        # ------- Prompt Generator -------
        self.promptgenerator = PromptGenerator_Encoder(
            n_channels=in_channels,
            n_classes=num_classes,
            normalization=normalization,
            has_dropout=has_dropout,
            pretrain_weight_path=pretrain_weight_path,
            num_points_per_class=num_points_per_class,
            threshold=threshold
        )

        # ------- Prompt Encoder -------
        self.promptencoder = PromptEncoder3D(
            embed_dim=embed_dim,
            image_embedding_size=self.embedding_size,     # e.g., (14, 14, 10)
            input_image_size=image_size,                  # e.g., (112, 112, 80)
            mask_in_chans=mask_in_chans,
            activation=activation
        )

        # ------- Image Encoder -------
        self.samencoder = ImageEncoderViT3D(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            out_chans=out_chans
        )

        # ------- Mask Decoder -------
        self.maskdecoder = MaskDecoder3D(
            num_multimask_outputs=num_multimask_outputs,
            transformer_dim=embed_dim,
            iou_head_depth=iou_head_depth
        )

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
        mask = self.postprocess_masks(after_maskencoder, input_size=x.shape[-3:], original_size=x.shape[-3:])

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
    output = model(input_tensor)
    print(f'output shape is {output.shape}')
    # summary(model=model, input_size=(1, 1, 112, 112, 80))

    return


if __name__ == "__main__":
    # test_imageencoder_vit3d()
    # PromptEncoder_Test_Full()
    # PromptEncoder_Test_Pic()
    networktest()
