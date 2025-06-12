"""
v14_2 dev
在V14_1的基础上进行改进
优化代码的内容
同时调整一些参数，让网络效果更好
另外采取一些新的训练模式
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

# 导入不确定性VNet网络
from networks.VNet_MultiOutput import VNet

# 导入解耦不确定性Prompt部分
from networks.DecoupledUncertaintyTensorTest import DecoupledUncertaintyGenerator
from networks.DecoupledUncertaintyTensorTest import find_top_k_prompts_per_class

# 导入可以生成明显背景和前景的张量的函数
from networks.DecoupledUncertaintyTensorTest import create_sphere_tensor

class DecoupledUncertaintyPrompt(nn.Module):
    """
    解耦不确定性点Prompt生成器 (已优化，无循环)
    """
    def __init__(self, n_channels: int, 
                 n_classes: int, 
                 normalization: Optional[str] = 'batchnorm',
                 has_dropout: bool = True, 
                 pretrain_weight_path: Optional[str] = None,
                 num_mc_samples=10, 
                 w_epistemic: float = 0.5, 
                 w_aleatoric: float = 0.5, 
                 k_per_class=20,
                 suppression_radius=10):
        super().__init__()
        
        # 注意初始化顺序
        self.network = VNet(n_channels=n_channels, n_classes=n_classes,
                            normalization=normalization, has_dropout=has_dropout)
        if pretrain_weight_path:
                    self.load_model(self.network, pretrain_weight_path)
        self.w_epistemic = w_epistemic
        self.w_aleatoric = w_aleatoric
        self.num_mc_samples = num_mc_samples
        self.num_classes = n_classes
        self.k_per_class = k_per_class
        self.suppression_radius = suppression_radius
        
        # 加载不确定性生成器
        self.uncertainty_gen = DecoupledUncertaintyGenerator(network=self.network,
                                                             num_mc_samples=self.num_mc_samples)
    @staticmethod
    def _normalize_batch(tensor: torch.Tensor) -> torch.Tensor:
        """ 对批次中的每个样本独立进行min-max归一化 """
        B = tensor.shape[0]
        # 保持维度以进行广播
        t_min = tensor.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1, 1)
        t_max = tensor.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1, 1)
        return (tensor - t_min) / (t_max - t_min + 1e-8)
    
    def _batched_nms_topk(self, scores: torch.Tensor, seg_probs: torch.Tensor):

        B, C, D, H, W = seg_probs.shape
        K = self.k_per_class
        
        # 1. 对每个类别，用 suitability score 乘以其概率，得到特定类别的 score map
        #    scores: [B, 1, D, H, W] -> 广播到 [B, C, D, H, W]
        class_specific_scores = scores * seg_probs
        # 2. 使用 3D 最大池化进行非极大值抑制 (NMS)
        #    一个点是局部最大值，当且仅当它的值等于该区域最大池化后的值
        kernel_size = self.suppression_radius * 2 + 1
        pooled_scores = F.max_pool3d(class_specific_scores, kernel_size, stride=1, padding=self.suppression_radius)
        local_maxima_mask = (class_specific_scores == pooled_scores)
        
        # 将非局部最大值的点分数置为0
        suppressed_scores = class_specific_scores * local_maxima_mask
        
        # 3. 选取 Top-K
        # 将空间维度展平，以便使用 topk
        suppressed_scores_flat = suppressed_scores.view(B, C, -1)
        
        # 在每个样本的每个类别中，独立寻找 top-k 的分数和索引
        # topk_scores/indices shape: [B, C, K]
        _, topk_indices_flat = torch.topk(suppressed_scores_flat, K, dim=-1, sorted=False)
        # 4. 将扁平化的索引转换回 3D 坐标
        # 创建一个与索引形状匹配的类别标签张量
        # class_ids shape: [B, C, K]
        class_ids = torch.arange(C, device=scores.device).view(1, C, 1).expand(B, C, K)
        
        # 坐标转换公式
        # d = index // (H * W)
        # h = (index % (H * W)) // W
        # w = index % W
        coords_d = topk_indices_flat // (H * W)
        coords_h = (topk_indices_flat % (H * W)) // W
        coords_w = topk_indices_flat % W
        
        # 将坐标堆叠起来，形状: [B, C, K, 3]
        topk_coords = torch.stack([coords_d, coords_h, coords_w], dim=-1)
        
        # 5. 整理成最终输出格式
        # 展平类别和K维度
        # all_coords shape: [B, C*K, 3]
        all_coords = topk_coords.reshape(B, C * K, 3)  # 修改为reshape
        # all_labels shape: [B, C*K]
        all_labels = class_ids.reshape(B, C * K)  # 修改为reshape
        
        return all_coords.float(), all_labels.long()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. 生成不确定性和分割概率
        H_epistemic, H_aleatoric, final_segmentation_probs = self.uncertainty_gen(x)
            
        # 2. 归一化不确定性 (批处理方式)
        H_epistemic_norm = self._normalize_batch(H_epistemic)
        H_aleatoric_norm = self._normalize_batch(H_aleatoric)
        
        # 3. 计算 Prompt Suitability Score
        # 形状: [B, 1, D, H, W]
        prompt_suitability_score = self.w_epistemic * (1 - H_epistemic_norm) + self.w_aleatoric * (1 - H_aleatoric_norm)
        
        # 4. 使用批处理方式并行计算所有样本的 prompts
        # a. 确定每个体素最可能的类别
        pred_classes_probs, pred_classes_idx = torch.max(final_segmentation_probs, dim=1, keepdim=True)
        # b. 为每个类别创建 one-hot mask
        # 注意：这里我们使用原始概率，而不是one-hot，这样可以保留更多信息
        # 如果需要硬性 one-hot，可以使用 F.one_hot(pred_classes_idx.squeeze(1), num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
        
        # 5. 调用批处理函数获取坐标和标签
        all_coords, all_labels = self._batched_nms_topk(prompt_suitability_score, final_segmentation_probs)
        
        return all_coords, all_labels
    
    # load_model 静态方法保持不变
    @staticmethod
    def load_model(model, model_path, device=None):
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

class PatchMerging3D(nn.Module):
    def __init__(self, input_dim: int, out_dim: Optional[int] = None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim or input_dim * 2  # 通常翻倍
        self.reduction = nn.Linear(input_dim * 8, self.out_dim, bias=False)
        self.norm = norm_layer(input_dim * 8)

    def forward(self, x):
        """
        x: [B, D, H, W, C]
        """
        B, D, H, W, C = x.shape
        # Pad if necessary
        if D % 2 == 1 or H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 0::2, 0::2, 1::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 1::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 1::2, 1::2, 0::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7],
                      dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class ImageEncoderViT3D(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = (112, 112, 80),  # 支持动态尺寸
        patch_size: int = 8,
        embed_dim: int = 192,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
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

        if isinstance(img_size, int):
            img_size = (img_size, img_size, img_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_embed1 = PatchEmbed3D(
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
            in_chans=1,
            embed_dim=192
        )

        # 动态位置编码
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    img_size[0] // 2,  # D
                    img_size[1] // 2,  # H
                    img_size[2] // 2,  # W
                    embed_dim
                )
            )

        # 设置其他规格的block进行尝试
        self.block1 = Block3D(
            dim=192,  # 第一个block的维度为192
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=4,  # 如果不设置window_size会进行全局注意力，导致显存爆炸
            input_size=(
                img_size[0] // 2,
                img_size[1] // 2,
                img_size[2] // 2,
            ),
        )

        self.block2 = Block3D(
            dim=384,  # 第一个block的维度为192
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=4,  # 如果不设置window_size会进行全局注意力，导致显存爆炸
            input_size=(
                img_size[0] // 4,
                img_size[1] // 4,
                img_size[2] // 4,
            )
        )

        self.block3 = Block3D(
            dim=768,  # 第一个block的维度为192
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=4,  # 如果不设置window_size会进行全局注意力，导致显存爆炸
            input_size=(
                img_size[0] // 4,
                img_size[1] // 4,
                img_size[2] // 4,
            )
        )

        self.block4 = Block3D(
            dim=768,  # 第一个block的维度为192
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=4,  # 如果不设置window_size会进行全局注意力，导致显存爆炸
            input_size=(
                img_size[0] // 4,
                img_size[1] // 4,
                img_size[2] // 4,
            )
        )


        self.patchmerging1 = PatchMerging3D(input_dim=192, out_dim=384)
        self.patchmerging2 = PatchMerging3D(input_dim=384, out_dim=768)
        self.patchmerging3 = PatchMerging3D(input_dim=768, out_dim=768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_list = []

        # 检查输入尺寸合法性
        assert x.ndim == 5, f"Input must be 5D [B,C,D,H,W], got {x.shape}"
        assert all([s % self.patch_size == 0 for s in x.shape[-3:]]), \
            f"Input spatial size {x.shape[-3:]} must be divisible by patch_size {self.patch_size}"

        # Patch Embedding
        x = self.patch_embed1(x)

        # 添加位置编码
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # 通过第一个Block
        x = x.permute(0, 3, 1, 2, 4)
        x = self.block1(x)

        # Patchmerging操作
        x = self.patchmerging1(x)

        # 通过第二个Block
        x = self.block2(x)

        # 通过PatchMerging
        x = self.patchmerging2(x)

        # 通过第三个Block
        x = self.block3(x)

        # 通过第四个Block
        x = self.block4(x)

        x = x.permute(0,4,2,3,1)


        return x

class Network(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            image_size: Tuple[int, int, int] = (112, 112, 80),
            patch_size: int = 8,
            embed_dim: int = 768,
            out_chans: int = 768,
            num_classes: int = 2,
            normalization: str = "batchnorm",
            has_dropout: bool = True,
            pretrain_weight_path: str = "result/VNet_Multi/LA_16/Pth/Best.pth",
            num_mc_samples:int = 5,
            w_epistemic:float=0.5,
            w_aleatoric:float=0.5,
            k_per_class:int=20,
            suppression_radius:int=7,
            mask_in_chans: int = 16,
            activation=nn.GELU,
            num_multimask_outputs: int = 2,
            iou_head_depth: int = 3,
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
        self.out_chans = out_chans

        # ------- PromptGenerator参数 -------
        self.pretrain_weight_path = pretrain_weight_path
        self.num_mc_samples = num_mc_samples
        self.w_epistemic = w_epistemic
        self.w_aleatoric = w_aleatoric
        self.k_per_class = k_per_class
        self.suppression_radius = suppression_radius

        # ------- PromptEncoder参数 -------
        self.embedding_size = tuple(s // patch_size for s in image_size)  # e.g., 112 -> 14
        self.mask_in_chans = mask_in_chans
        self.activation = activation

        # ------- MaskDecoder参数 -------
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth

        # ------- Prompt Generator -------
        self.promptgenerator = DecoupledUncertaintyPrompt(
            n_channels=self.in_channels,        # 输入通道数
            n_classes=self.n_classes,              # 输出类别数
            normalization=self.normalization,
            has_dropout=self.has_dropout,
            pretrain_weight_path=self.pretrain_weight_path,
            num_mc_samples=self.num_mc_samples,
            w_epistemic=self.w_epistemic,
            w_aleatoric=self.w_aleatoric,
            k_per_class=self.k_per_class,
            suppression_radius = self.suppression_radius
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
            embed_dim=192
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
        vnet_output, variance = self.vnet(x)
        
        # 1. 图像主干编码
        after_encoder = self.samencoder(x)
        
        # 2. 获得prompt（返回固定维度的张量）
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
        
        return vnet_output, sam_output

def networktest():
    # 检查CUDA是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU")

    # 实例化网络
    model = Network(in_channels=1).to(device=device)

    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device=device)
    model(input_tensor) # 向前传播
    summary(model, input_size=(1, 1, 112, 112, 80), device=device)

    return 

if __name__ == "__main__":
    networktest()
