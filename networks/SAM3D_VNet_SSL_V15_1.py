"""
v15_1
优化网络结构
另外让网络的参数变少，同时提高网络的可解释性
"""

import sys
from turtle import forward
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

# 导入第二代不确定性VNet网络
# from networks.VNet_MultiOutput import VNet
# from networks.VNet_MultiOutput_V2 import VNet
from networks.VNet_MultiOutput_V3 import VNet

# 导入解耦不确定性Prompt部分
from networks.DecoupledUncertaintyTensorTest import DecoupledUncertaintyGenerator
from networks.DecoupledUncertaintyTensorTest import find_top_k_prompts_per_class

# 导入可以生成明显背景和前景的张量的函数
from networks.DecoupledUncertaintyTensorTest import create_sphere_tensor

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

def pathmerging3D_test():
    # 检查CUDA是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU")

    # 实例化PatchMerging3D
    patch_merging = PatchMerging3D(input_dim=256, out_dim=512).to(device)

    # 模拟输入数据
    input_tensor = torch.randn(1, 14, 14, 10, 256).to(device)  # 假设输入尺寸为 [B, D, H, W, C]

    output_tensor = patch_merging(input_tensor)
    
    print(f"输出张量形状: {output_tensor.shape}")
    
    return output_tensor

class ImageEncoderViT3D(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = (112, 112, 80),  # 支持动态尺寸
        patch_size: int = 8,
        embed_dim: int = 192,
        num_heads: int = 12,
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
            dim=192,  
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
            dim=384,  
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
            dim=768,  
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
            dim=768,  
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

        # patchmerging方式
        self.patchmerging1 = PatchMerging3D(input_dim=192, out_dim=384)
        self.patchmerging2 = PatchMerging3D(input_dim=384, out_dim=768)
        self.patchmerging3 = PatchMerging3D(input_dim=768, out_dim=768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

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

        # 输出结果调整维度
        x = x.permute(0,4,2,3,1)

        return x

class CARF(nn.Module):
    def __init__(self, dim_a, dim_b, common_dim=256):
        super().__init__()
        
        # 投影层，将两个输入特征投影到相同的维度
        self.proj_a = nn.Conv3d(dim_a, common_dim, kernel_size=1, bias=False)
        self.proj_b = nn.Conv3d(dim_b, common_dim, kernel_size=1, bias=False)
        
        # 最终融合后的处理层
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(common_dim, common_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(common_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature_a, feature_b):
        # --- 1. 输入格式统一 ---
        # 验证输入维度
        assert feature_a.dim() == 5, f"Feature A must be 5D, but got {feature_a.dim()}D"
        assert feature_b.dim() == 5, f"Feature B must be 5D, but got {feature_b.dim()}D"
        
        # 将输入特征从 (B, C, H, W, D) 转换为 PyTorch 标准的 (B, C, D, H, W)
        # torch.Size([B, C, H, W, D]) -> torch.Size([B, C, D, H, W])
        f_a_std = feature_a.permute(0, 1, 4, 2, 3) 
        f_b_std = feature_b.permute(0, 1, 4, 2, 3) # <--- 这是关键的修正点
        
        # print(f'Permuted feature_a shape (BCDHW): {f_a_std.shape}')
        # print(f'Permuted feature_b shape (BCDHW): {f_b_std.shape}')

        # --- 2. 共享维度投影 ---
        # f_a_proj, f_b_proj 的形状都将是 (B, common_dim, D, H, W)
        f_a_proj = self.proj_a(f_a_std)
        f_b_proj = self.proj_b(f_b_std)
        
        # print(f'Projected f_a_proj shape: {f_a_proj.shape}')
        # print(f'Projected f_b_proj shape: {f_b_proj.shape}')

        # --- 3. 注意力图生成 ---
        # 沿通道维度计算余弦相似度，得到每个空间位置的相似度分数
        # sim_map 的形状是 (B, D, H, W)
        sim_map = F.cosine_similarity(f_a_proj, f_b_proj, dim=1)
        
        # 使用 sigmoid 将相似度 (-1, 1) 映射到 (0, 1) 作为门控权重
        # attention_map 形状 (B, 1, D, H, W) 以便进行广播乘法
        attention_map = torch.sigmoid(sim_map).unsqueeze(1)
        
        # print(f'Attention map shape: {attention_map.shape}')

        # --- 4. 双向特征精炼 ---
        # attention_map 接近 1 的区域表示 A 和 B 的特征在此处相似
        # f_a_refined: 保留编码器A特征中与B相似的部分
        f_a_refined = f_a_proj * attention_map
        
        # (1 - attention_map) 接近 1 的区域表示 A 和 B 的特征在此处不相似（互补）
        # f_b_refined: 保留编码器B特征中与A互补的部分
        f_b_refined = f_b_proj * (1 - attention_map)
        
        # --- 5. 最终融合 ---
        # 将 A 的相似部分与 B 的互补部分相加，实现智能融合
        fused_feature = f_a_refined + f_b_refined
        
        # 通过最后的卷积层进一步处理融合特征，增强表达能力
        fused_feature = self.fusion_conv(fused_feature)

        # 调整维度顺序
        fused_feature = fused_feature.permute(0,1,3,4,2)
        
        # print(f'Final fused feature shape: {fused_feature.shape}')
        
        # 返回融合后的特征和去掉通道维度的注意力图
        return fused_feature, attention_map.squeeze(1)

def CARF_test():
    # 1. 根据你的输入形状定义参数
    batch_size = 1
    dim_a = 256  # feature_a 的通道数
    dim_b = 768  # feature_b 的通道数
    common_dim = 768 # 投影到的公共维度
    H, W, D = 14, 14, 10

    # 2. 实例化模型
    model = CARF(dim_a=dim_a, dim_b=dim_b, common_dim=common_dim)
    print("CARF model instantiated.")
    
    # 3. 创建与你描述形状一致的伪数据
    feature_a = torch.randn(batch_size, dim_a, H, W, D)
    feature_b = torch.randn(batch_size, dim_b, H, W, D)
    
    print(f"\nInput feature_a shape: {feature_a.shape}") # (1, 256, 14, 14, 10)
    print(f"Input feature_b shape: {feature_b.shape}") # (1, 768, 14, 14, 10)

    # 4. 将伪数据输入模型
    # 开启 torch.no_grad() 因为我们只是在做前向传播测试，不是在训练
    with torch.no_grad():
        fused_output, attention_map_output = model(feature_a, feature_b)

    # 5. 打印输出形状以验证
    print("\n--- Output Shapes ---")
    # 最终输出特征的形状应为 (B, common_dim, D, H, W)
    print(f"Fused feature output shape: {fused_output.shape}") 
    # 注意力图的形状应为 (B, D, H, W)
    print(f"Attention map output shape: {attention_map_output.shape}")

    # --- 预期的输出 ---
    # Fused feature output shape: torch.Size([1, 256, 10, 14, 14])
    # Attention map output shape: torch.Size([1, 10, 14, 14])

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
            # pretrain_weight_path: str = "./result/VNet_Multi_V3/Amos_90/Pth/Best.pth",
            pretrain_weight_path: str = None,
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

        # ------- VNet_MultiOutput_V3 -------
        self.vnet = VNet(n_channels=self.in_channels,
                         n_classes=self.n_classes,
                         normalization=self.normalization,
                         n_filters=32,
                         has_dropout=self.has_dropout)

        if self.pretrain_weight_path:
            # print(f'Loading pretrained weights from {self.pretrain_weight_path}')
            self.vnet = self.load_model(self.vnet, self.pretrain_weight_path)

        # ------- upsample -------
        self.upsampler = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=num_multimask_outputs,  # 2
                out_channels=num_multimask_outputs,  # 保持通道数不变
                kernel_size=3,
                stride=2,  # 2倍上采样
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm3d(num_multimask_outputs),
            nn.ReLU(inplace=True),
        )

        # 加载不确定性生成器
        self.uncertainty_gen = DecoupledUncertaintyGenerator(network=self.vnet,
                                                             num_mc_samples=self.num_mc_samples)

        # multi encoder merge
        self.carf_module = CARF(dim_a=256, dim_b=768, common_dim=768)

    @staticmethod
    def _normalize_batch(tensor: torch.Tensor) -> torch.Tensor:
        """ 对批次中的每个样本独立进行min-max归一化 """
        B = tensor.shape[0]
        # 保持维度以进行广播
        t_min = tensor.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1, 1)
        t_max = tensor.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1, 1)
        return (tensor - t_min) / (t_max - t_min + 1e-8)
    
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
        class_ids = torch.arange(C, device=scores.device).view(1, C, 1).expand(B, C, K)
        
        # 坐标转换
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
    
    def DecoupledUncertaintyPrompt_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1. 生成不确定性和分割概率
        H_epistemic, H_aleatoric, final_segmentation_probs = self.uncertainty_gen(x)

        # 2. 归一化不确定性 (批处理方式)
        H_epistemic_norm = self._normalize_batch(H_epistemic)
        H_aleatoric_norm = self._normalize_batch(H_aleatoric)
        
        # 3. 计算 Prompt Suitability Score
        prompt_suitability_score = self.w_epistemic * (1 - H_epistemic_norm) + self.w_aleatoric * (1 - H_aleatoric_norm)
        
        # 4. 使用批处理方式并行计算所有样本的 prompts
        pred_classes_probs, pred_classes_idx = torch.max(final_segmentation_probs, dim=1, keepdim=True)
        
        # 5. 调用批处理函数获取坐标和标签
        all_coords, all_labels = self._batched_nms_topk(prompt_suitability_score, final_segmentation_probs)
        
        return all_coords, all_labels

    def forward(self, x):
        # 0. VNet输出结果
        vnet_output, variance, encoder_feature = self.vnet(x)
        print(f'vnet_output shape is {vnet_output.shape}')
        print(f'variance shape is {variance.shape}')
        for i in range(len(encoder_feature)):
            print(f'encoder_feature[{i}] shape is {encoder_feature[i].shape}')

        # 1. 图像主干编码
        after_encoder = self.samencoder(x)

        # 2. 获得prompt
        coords, labels = self.DecoupledUncertaintyPrompt_forward(x)
        # print(f'coords shape is {coords.shape}, labels shape is {labels.shape}')
        
        # 3. prompt 编码
        sparse_embeddings, dense_embeddings = self.promptencoder(
            points=(coords, labels),
            boxes=None,
            masks=None
        )
        # print(f'sparse_embeddings shape is {sparse_embeddings.shape}')
        # print(f'dense_embeddings shape is {dense_embeddings.shape}')

        # 4. 获得位置编码
        image_pe = self.promptencoder.get_dense_pe()
        # print(f'image_pe shape is {image_pe.shape}')
        
        # 5. encoder output合并
        dual_encoder_output = self.carf_module(encoder_feature[3],after_encoder)
        dual_encoder_output_part1 = dual_encoder_output[0]
        # print(f'dual_encoder_output shape is {dual_encoder_output_part1.shape}')

        # 4. 掩码解码
        after_maskencoder, iou_pred = self.maskdecoder(
            image_embeddings=dual_encoder_output_part1,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True
        )
        # print(f'after maskencoder shape is {after_maskencoder.shape}')
        
        # 5. 插值还原尺寸
        sam_output = self.upsampler(after_maskencoder)
        # print(f'sam_output shape is {sam_output.shape}')
        
        return vnet_output, sam_output

def networktest():
    # 检查CUDA是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU")
    # 实例化网络
    model = Network(in_channels=1,num_classes=2).to(device=device)

    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device=device) # LA数据集模拟输入大小
    # input_tensor = torch.randn(1,1,160,160,80).to(device=device) # Amos数据集的输入大小

    model(input_tensor) # 向前传播

    # 通过summary计算模型复杂度
    # summary(model, input_size=(1, 1, 112, 112, 80), device=device)

    return 

if __name__ == "__main__":
    networktest()
    # CARF_test()