"""
v11
Depth保持8不变
想办法实现点Prompt在每一层都要有10个
一共是800个点
然后
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
    用于从粗分割（概率图）中生成点Prompt。
    修改后的版本：在每个 Z 轴切片上采样指定数量的点。
    """
    def __init__(
        self,
        n_channels: int, # 图像通道数量
        n_classes: int, # 分割的类别数
        normalization: Optional[str] = None, # VNet的归一化类型
        has_dropout: bool = False, # VNet是否使用Dropout
        pretrain_weight_path: Optional[str] = None, # VNet的预训练权重
        num_points_per_slice: int = 10, # 现在是每层采样点数
        threshold: float = 0.5,
        sample_mode: str = "random",
        debug: bool = False,
    ):
        super().__init__()

        # 检查 threshold 和 num_points_per_slice 参数是否合理
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold 必须在 [0, 1] 范围内, 得到 {threshold}")
        if num_points_per_slice <= 0:
            raise ValueError(f"num_points_per_slice 必须大于 0, 得到 {num_points_per_slice}")
        assert sample_mode in ("random", "topk"), "sample_mode 必须是 'random' 或 'topk'"

        # 使用 VNet 进行分割
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
                print(f'从 {pretrain_weight_path} 加载预训练权重')
            self.load_model(model=self.network, model_path=pretrain_weight_path)

        self.num_points_per_slice = num_points_per_slice # 存储每片采样点数
        self.threshold = threshold
        self.sample_mode = sample_mode
        self.debug = debug


    @staticmethod
    def load_model(model, model_path, device=None):
        """
        加载模型权重，自动处理单卡/多卡模型保存的权重。
        """
        state_dict = torch.load(model_path, map_location=device)
        # 检查是否是 DataParallel 保存的模型
        if any(key.startswith('module.') for key in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # 移除 'module.' 前缀
                new_state_dict[name] = v
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
        从单通道概率图 prob_map (形状 = spatial_dims) 中，
        找到概率 > threshold 的所有坐标，并从中采样 num_points 个点。
        支持随机采样 ('random') 或 top-k 概率采样 ('topk')。

        Args:
            prob_map (torch.Tensor): 输入的概率图 (例如 2D 切片 D*H 或 3D 体积 D*H*W)。
            label_value (int): 这些采样点应分配的标签值。
            num_points (int): 要采样的目标点数。
            threshold (float): 概率阈值。
            mode (str): 采样模式 ('random' 或 'topk')。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - sampled_coords (torch.Tensor): 采样点的坐标 (M, spatial_dim)，M <= num_points。
                - sampled_labels (torch.Tensor): 采样点的标签 (M,)，所有值都为 label_value。
        """
        # 找到所有概率大于阈值的点的坐标
        # nonzero 返回一个 (K, spatial_dim) 的张量，K 是满足条件的点的数量
        coords_all = (prob_map > threshold).nonzero(as_tuple=False) # (K, spatial_dim)
        K = coords_all.size(0)

        # 如果没有点满足阈值条件
        if K == 0:
            return (
                torch.empty(0, prob_map.dim(), device=prob_map.device, dtype=torch.long), # 返回空坐标张量
                torch.empty(0, dtype=torch.long, device=prob_map.device),                 # 返回空标签张量
            )

        # 确定实际要采样的点数 M，不能超过候选点总数 K
        M = min(num_points, K)

        if mode == "random":
            # 随机采样：生成 0 到 K-1 的随机排列，取前 M 个索引
            idx = torch.randperm(K, device=prob_map.device)[:M]

        elif mode == "topk":
            # Top-k 采样：获取满足条件点的概率值
            # coords_all.unbind(dim=1) 将 (K, spatial_dim) 拆分成 spatial_dim 个 (K,) 的张量
            # prob_map[...] 使用这些索引来获取对应的概率值
            values = prob_map[coords_all.unbind(dim=1)]
            # 找到概率最高的 M 个点的索引
            _, topk_idx = torch.topk(values, M)
            idx = topk_idx
        else:
            raise ValueError(f"不支持的采样模式: {mode}")

        # 根据选定的索引 idx 获取最终的采样点坐标
        sampled_coords = coords_all[idx]  # (M, spatial_dim)
        # 创建对应的标签张量，所有标签都设为 label_value
        sampled_labels = torch.full((M,), label_value, dtype=torch.long, device=prob_map.device)

        return sampled_coords, sampled_labels

    def _extract_point_prompts(
        self,
        logits: torch.Tensor,
        num_points_per_slice: int, # 参数更新
        threshold: float,
        sample_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C = logits.shape[:2]         # B: batch size, C: 类别数
        spatial_dims = logits.shape[2:] # 获取空间维度HWD
        spatial_dim_count = len(spatial_dims) # 空间维度的数量 (例如 3D 为 3)
        device = logits.device

        # 检查维度
        if spatial_dim_count != 3:
             raise ValueError(f"期望 3D 空间维度 (例如 D, H, Z)，但得到 {spatial_dim_count} 维: {spatial_dims}")
        
        num_slices = spatial_dims[-1] # 层数

        if self.debug:
            print(f'---------------------Debug信息----------------------------')
            print(f"输入 logits 形状: {logits.shape}")
            print(f"空间维度: {spatial_dims}, 切片数 (假设是最后一维): {num_slices}")
            print(f"每个类别每片采样点数: {num_points_per_slice}")
            print(f'---------------------Debug信息----------------------------')

        # 计算概率图
        probs = F.softmax(logits, dim=1) # 在类别维度上计算 softmax

        batch_coords: List[torch.Tensor] = [] # 存储每个 batch 样本的坐标
        batch_labels: List[torch.Tensor] = [] # 存储每个 batch 样本的标签

        # 遍历 batch 中的每个样本
        for b in range(B):
            coords_per_sample: List[torch.Tensor] = [] # 当前样本收集的所有坐标
            labels_per_sample: List[torch.Tensor] = [] # 当前样本收集的所有标签

            # 遍历每个前景类别
            for c in range(1, C):
                if self.debug:
                    print(f'\n[样本 {b+1}/{B}, 类别 {c}] 处理中...')

                coords_per_class_sample: List[torch.Tensor] = [] # 当前类别当前样本的所有切片坐标
                labels_per_class_sample: List[torch.Tensor] = [] # 当前类别当前样本的所有切片标签

                # 获取当前类别 c 的概率图，形状HWD
                prob_map_3d = probs[b, c]

                # 遍历每个切片
                for z in range(num_slices):
                    # 获取当前 2D 切片 (形状: [D, H])
                    prob_map_slice = prob_map_3d[..., z] # 取最后一个维度

                    # 在当前 2D 切片上采样点
                    coords_2d, labels_slice = self._sample_points_from_mask(
                        prob_map=prob_map_slice,
                        label_value=c,
                        num_points=num_points_per_slice, # 使用每片采样数
                        threshold=threshold,
                        mode=sample_mode,
                    )

                    # 如果在当前切片上采样到了点，那么把2D坐标恢复成3D坐标
                    if coords_2d.numel() > 0:
                        z_coords = torch.full_like(labels_slice, z, dtype=torch.long).unsqueeze(1)
                        coords_3d = torch.cat((coords_2d, z_coords), dim=1)

                        # 收集当前切片的 3D 坐标和标签
                        coords_per_class_sample.append(coords_3d)
                        labels_per_class_sample.append(labels_slice)

                # 如果当前类别在该样本的所有切片中采样到了点
                if coords_per_class_sample:
                    # 将该类别的所有点连接起来
                    coords_cat_class = torch.cat(coords_per_class_sample, dim=0)
                    labels_cat_class = torch.cat(labels_per_class_sample, dim=0)
                    if self.debug:
                         print(f'[样本 {b+1}/{B}, 类别 {c}] 总共采样到 {coords_cat_class.shape[0]} 个点')
                    # 添加到当前样本的总列表中
                    coords_per_sample.append(coords_cat_class)
                    labels_per_sample.append(labels_cat_class)
                elif self.debug:
                     print(f'[样本 {b+1}/{B}, 类别 {c}] 未采样到任何点')


            if self.debug:
                 print(f'\n[样本 {b+1}/{B}, 背景类别 0] 处理中...')
            coords_bg_sample: List[torch.Tensor] = []
            labels_bg_sample: List[torch.Tensor] = []
            prob_map_bg_3d = probs[b, 0] # 背景概率图

            for z in range(num_slices):
                prob_map_bg_slice = prob_map_bg_3d[..., z]
                coords_bg_2d, labels_bg_slice = self._sample_points_from_mask(
                    prob_map=prob_map_bg_slice,
                    label_value=0, # 背景标签为 0
                    num_points=num_points_per_slice,
                    threshold=threshold,
                    mode=sample_mode,
                )
                if coords_bg_2d.numel() > 0:
                    z_coords = torch.full_like(labels_bg_slice, z, dtype=torch.long).unsqueeze(1)
                    coords_bg_3d = torch.cat((coords_bg_2d, z_coords), dim=1)
                    coords_bg_sample.append(coords_bg_3d)
                    labels_bg_sample.append(labels_bg_slice)
                    # if self.debug:
                    #     print(f'  切片 {z}: 采样到 {coords_bg_3d.shape[0]} 个背景点')

            if coords_bg_sample:
                coords_cat_bg = torch.cat(coords_bg_sample, dim=0)
                labels_cat_bg = torch.cat(labels_bg_sample, dim=0)
                if self.debug:
                     print(f'[样本 {b+1}/{B}, 背景类别 0] 总共采样到 {coords_cat_bg.shape[0]} 个点')
                coords_per_sample.append(coords_cat_bg)
                labels_per_sample.append(labels_cat_bg)
            elif self.debug:
                print(f'[样本 {b+1}/{B}, 背景类别 0] 未采样到任何点')


            # --- 聚合当前样本的所有点 ---
            # 如果当前样本在所有类别、所有切片中都没有采样到任何点
            if not coords_per_sample:
                if self.debug:
                    print(f'[样本 {b+1}/{B}] 未采样到任何点，添加一个虚拟背景点')
                coords_per_sample = [
                    torch.zeros(1, spatial_dim_count, dtype=torch.long, device=device)
                ]
                labels_per_sample = [
                    torch.zeros(1, dtype=torch.long, device=device)
                ]

            # 将当前样本的所有类别的点连接起来
            coords_cat_sample = torch.cat(coords_per_sample, dim=0)
            labels_cat_sample = torch.cat(labels_per_sample, dim=0)

            if self.debug:
                print(f'\n[样本 {b+1}/{B}] 总计采样点数: {coords_cat_sample.shape[0]}')

            # 添加到 batch 列表中
            batch_coords.append(coords_cat_sample)
            batch_labels.append(labels_cat_sample)

        # --- 对 Batch 中的样本进行填充 (Padding) ---
        # 使 batch 中每个样本的点数相同，方便后续处理
        # 使用 pad_sequence：
        #   - batch_first=True 使输出形状为 (B, max_len, *)
        #   - padding_value=0 用于坐标 (虽然理想是-1，但坐标通常非负，0可能也行，取决于后续处理)
        #   - padding_value=-1 用于标签，表示这是一个填充的无效标签
        coords_padded = pad_sequence(batch_coords, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(batch_labels, batch_first=True, padding_value=-1) # 使用 -1 标记填充标签

        return coords_padded, labels_padded

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 检查输入维度是否正确
        if x.dim() < 5:
            raise ValueError(f"输入张量应至少为 5D (B, C, spatial...), 得到 {x.shape}")

        # 获取配置参数
        n_pts_slice = self.num_points_per_slice
        thr = self.threshold
        mode = self.sample_mode

        # 1. 获得粗分割结果
        logits = self.network(x)  # 输出形状: [B, num_classes, *spatial_dims]

        # 2. 从 logits 提取点 prompts
        coords, labels = self._extract_point_prompts(logits, n_pts_slice, thr, mode)

        return coords, labels

# ==========================================================
# 测试代码
# ==========================================================
def PromptGenerator_test():
    print("--- 开始 PromptGenerator 测试 ---")

    input_shape = (5, 1, 112, 112, 80) # BCHWD
    n_classes = 2                        # 2个类别: 0=背景, 1=前景
    num_points_per_slice = 5            # 每个类别在每个 Z 切片上采样 10 个点
    threshold = 0.5                      # 概率阈值
    sample_mode = 'random'                 # 采样模式 ('random' 或 'topk')
    debug_mode = True                    # 开启调试信息
    normalization_type = 'batchnorm'     # VNet 的归一化类型

    print(f"测试参数:")
    print(f"  输入形状: {input_shape}")
    print(f"  类别数: {n_classes}")
    print(f"  每片采样点数: {num_points_per_slice}")
    print(f"  阈值: {threshold}")
    print(f"  采样模式: {sample_mode}")
    print(f"  调试模式: {debug_mode}")

    # --- 创建模拟输入 ---
    mock_input = torch.rand(input_shape)
    print(f"\n创建模拟输入张量，形状: {mock_input.shape}")

    # --- 实例化模型 ---
    print("\n实例化 PromptGenerator_Encoder 模型...")
    model = PromptGenerator_Encoder(
        n_channels=input_shape[1],        # 输入通道数 = 1
        n_classes=n_classes,              # 输出类别数 = 2
        normalization=normalization_type,
        has_dropout=False,
        pretrain_weight_path=None,        # 不加载预训练权重
        num_points_per_slice=num_points_per_slice,
        threshold=threshold,
        sample_mode=sample_mode,
        debug=debug_mode
    )
    print("模型实例化完成。")

    with torch.no_grad(): # 在评估时不计算梯度
        coords_output, labels_output = model(mock_input)

    # --- 打印输出结果 ---
    print("\n--- 测试结果 ---")
    print(f"输出坐标形状 (coords_output.shape): {coords_output.shape}")
    print(f"输出标签形状 (labels_output.shape): {labels_output.shape}")

    # 分析预期点数
    num_slices = input_shape[-1] # D = 80
    max_expected_points_per_sample = n_classes * num_slices * num_points_per_slice
    print(f"\n理论上每个样本最多采样点数 (如果所有切片所有类别都满足条件):")
    print(f"  {n_classes} (类别数) * {num_slices} (切片数) * {num_points_per_slice} (每片点数) = {max_expected_points_per_sample}")
    print(f"实际采样点数 (输出张量的第二维): {coords_output.shape[1]}")
    print(f"(注意：实际点数会少于最大值，因为某些切片/类别的概率可能不满足阈值，或者候选点不足 {num_points_per_slice} 个)")

    # 打印第一个样本的部分采样点和标签
    if coords_output.shape[1] > 0: # 确保有采样点
        print("\n第一个样本的部分输出坐标 (前 5 个点):")
        print(coords_output[0, :5, :])
        print("\n第一个样本的部分输出标签 (前 5 个点):")
        print(labels_output[0, :5])

        # 检查坐标范围是否合理
        spatial_dims_shape = input_shape[2:] # (112, 112, 80)
        coords_np = coords_output[0].cpu().numpy() # 转到 CPU 并转为 NumPy
        labels_np = labels_output[0].cpu().numpy()
        valid_mask = labels_np != -1 # 排除填充点
        valid_coords = coords_np[valid_mask]

        if valid_coords.shape[0] > 0:
            min_coords = valid_coords.min(axis=0)
            max_coords = valid_coords.max(axis=0)
            print("\n第一个样本有效采样点坐标范围:")
            print(f"  维度 H (期望 [0, {spatial_dims_shape[0]-1}]): Min={min_coords[0]}, Max={max_coords[0]}")
            print(f"  维度 W (期望 [0, {spatial_dims_shape[1]-1}]): Min={min_coords[1]}, Max={max_coords[1]}")
            print(f"  维度 D (期望 [0, {spatial_dims_shape[2]-1}]): Min={min_coords[2]}, Max={max_coords[2]}")
            # 检查是否有坐标超出边界 (简单检查)
            if (min_coords < 0).any() or \
               (max_coords[0] >= spatial_dims_shape[0]) or \
               (max_coords[1] >= spatial_dims_shape[1]) or \
               (max_coords[2] >= spatial_dims_shape[2]):
                print("警告: 发现超出预期范围的坐标！")
            else:
                print("坐标范围看起来合理。")
        else:
            print("\n第一个样本没有采样到有效点（可能所有点都是填充点或虚拟点）。")

    else:
        print("\n输出张量中没有采样点。")

    print("\n--- 测试结束 ---")

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
            num_points_per_slice: int = 5, # 每个切片的Prompt的数量
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
        self.num_points_per_slice = num_points_per_slice # 添加切片参数
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
            num_points_per_slice=self.num_points_per_slice, # 添加切片中prompt数量
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
        # print(f'vnet output shape is {vnet_output.shape}')

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
    model = Network(in_channels=1,encoder_depth=4,num_points_per_slice=5).to(device=device)

    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device=device)
    vnet_output,sam_output = model(input_tensor)
    print(f"输出形状: {vnet_output.shape,sam_output.shape}")
    # summary(model, input_size=(1, 1, 112, 112, 80), device=device)

    return 


if __name__ == "__main__":
    networktest()
    # PromptGenerator_test()