"""
v13 dev

Tasks:
1. refine the ImageEncoder
"""

import sys
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Optional, Type, Union, List
import matplotlib.pyplot as plt
from torchinfo import summary

# 导入sam3D主要相关模块
from networks.sam_med3d.modeling.prompt_encoder3D import PromptEncoder3D
# from networks.sam_med3d.modeling.mask_decoder3D import MaskDecoder3D
from networks.sam_med3d.modeling.mask_decoder3D import MLP
from networks.sam_med3d.modeling.mask_decoder3D import TwoWayTransformer3D

# 导入ImageEncoder其他模块
from networks.sam_med3d.modeling.image_encoder3D import Block3D
from networks.sam_med3d.modeling.image_encoder3D import PatchEmbed3D
from networks.sam_med3d.modeling.image_encoder3D import LayerNorm3d

# 导入粗分割VNet网络
from networks.VNet import VNet

class BayesianProbabilityGenerator(nn.Module):
    def __init__(self, network, num_mc_samples=3):  # 进一步减少采样次数
        super().__init__()
        self.network = network
        self.num_mc_samples = num_mc_samples
        self._enable_dropout()

    def _enable_dropout(self):
        for m in self.network.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, x):
        # 初始化为与x相同设备和类型
        mean = torch.zeros_like(self.network(x), device=x.device)  # 初始化mean
        M2 = torch.zeros_like(mean)
        
        # 梯度计算开关（根据训练/测试模式）
        grad_enabled = torch.is_grad_enabled()
        
        for i in range(1, self.num_mc_samples + 1):
            with torch.set_grad_enabled(grad_enabled):  # 跟随主程序梯度设置
                # 前向传播时禁用自动求导的中间变量存储
                with torch.no_grad():  # 关键优化点1：禁用梯度计算
                    logits = self.network(x)
                logits = logits.detach()  # 关键优化点2：分离计算图

                # Welford算法更新
                delta = logits - mean
                mean += delta / i
                delta2 = logits - mean
                M2 += delta * delta2

                # 及时释放中间变量
                del logits, delta, delta2

        # 计算方差
        var = M2 / (self.num_mc_samples - 1) if self.num_mc_samples > 1 else 0
        
        # 最终计算时恢复梯度
        with torch.set_grad_enabled(grad_enabled):
            calibrated_logits = mean / (1 + var.sqrt())
            probs = F.softmax(calibrated_logits, dim=1)
        
        return probs

def test_BayesianProbabilityGenerator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W, D = 1, 1, 112, 112, 80  # 输入形状
    n_classes = 2  # 假设分割任务为二分类
    num_mc_samples = 5  # 蒙特卡洛采样次数
    
    # 初始化网络
    vnet = VNet(n_channels=1,n_classes=2,normalization='batchnorm',has_dropout=True).to(device=device)
    model = BayesianProbabilityGenerator(vnet, num_mc_samples=num_mc_samples).to(device)
    model.eval()  # 测试时仍保持 Dropout 激活
    
    # -------------------- 测试案例1: 正常输入 --------------------
    print(f'进行正常输入测试')
    torch.manual_seed(42)
    x_normal = torch.randn(B, C, H, W, D, device=device)  # 模拟输入
    probs = model(x_normal)
    print(f'正常输入的输出大小为{probs.shape}')

    # 验证输出形状
    assert probs.shape == (B, n_classes, H, W, D), \
        f"概率图形状错误，应为 {(B, n_classes, H, W, D)}，实际为 {probs.shape}"
    
    # 验证概率值范围 [0,1]
    assert (probs >= 0).all() and (probs <= 1).all(), "概率值超出 [0,1] 范围"
    
    # 验证类别维度概率和为1
    prob_sum = probs.sum(dim=1)  # 沿类别维度求和
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-4), "类别概率和不为1"
    
    # -------------------- 测试案例2: 全零输入 --------------------
    print(f'进行全零输入测试')
    x_zeros = torch.zeros(B, C, H, W, D, device=device)
    probs_zeros = model(x_zeros)
    print(f'全零输入的输出大小为{probs_zeros.shape}')
    # 由于 Dropout 存在，输出不全为零
    assert not torch.allclose(probs_zeros, torch.zeros_like(probs_zeros)), "全零输入导致零概率"

    # 检查模型占用
    # summary(model=model, input_size=(B, C, H, W, D), device=device)

class PromptGenerator_Encoder(nn.Module):
    def __init__(
        self,
        n_channels: int,  # 图像通道数量
        n_classes: int,  # 分割的类别数
        normalization: Optional[str] = None,  # VNet的归一化类型
        has_dropout: bool = False,  # VNet是否使用Dropout
        pretrain_weight_path: Optional[str] = None,  # VNet的预训练权重
        num_points_per_slice: int = 10,  # 现在是每层采样点数
        threshold: float = 0.5,
        sample_mode: str = "random",
        debug: bool = False,
    ):
        super().__init__()

        # 检查 threshold 和 num_points_per_slice 参数是否合理
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold 必须在 [0, 1] 范围内, 得到 {threshold}")
        if num_points_per_slice <= 0:
            raise ValueError(
                f"num_points_per_slice 必须大于 0, 得到 {num_points_per_slice}")
        assert sample_mode in (
            "random", "topk","entropy"), "sample_mode 必须是 'random' 或 'topk'或者是'entropy'"

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
            self.load_model(model=self.network,
                            model_path=pretrain_weight_path)

        # 初始化BayesianProbabilityGenerator
        self.bayesian_generator = BayesianProbabilityGenerator(
            self.network, num_mc_samples=4)

        self.num_points_per_slice = num_points_per_slice  # 存储每片采样点数
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
                name = k[7:]  # 移除 'module.' 前缀
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

        # 找到满足阈值条件的所有坐标
        coords_all = (prob_map > threshold).nonzero(as_tuple=False)  # (K, spatial_dim)
        K = coords_all.size(0)

        # 若无满足条件的点
        if K == 0:
            return (
                torch.empty(0, prob_map.dim(), device=prob_map.device, dtype=torch.long),
                torch.empty(0, dtype=torch.long, device=prob_map.device),
            )
        
        # 确定实际采样点数
        M = min(num_points, K)
        
        if mode == "random":
            # 随机采样
            idx = torch.randperm(K, device=prob_map.device)[:M]
        elif mode == "topk":
            # 基于概率值的top-k采样
            values = prob_map[coords_all.unbind(dim=1)]
            _, topk_idx = torch.topk(values, M)
            idx = topk_idx
        elif mode == "entropy":
            if prob_map.dim() < 2:
                raise ValueError("熵模式需要包含类别概率的prob_map。确保prob_map形状为(C, spatial_dims)。")
            # 重塑为(C, N)格式（N为空间点数）
            C, *spatial_dims = prob_map.shape
            prob_map_flat = prob_map.view(C, -1)  # (C, N)
            # 计算每个点的熵
            entropy = -torch.sum(prob_map_flat * torch.log(prob_map_flat + 1e-8), dim=0)  # (N,)
            # 仅考虑高于阈值的点
            entropy_above_threshold = entropy[coords_all[:, 0]]
            _, topk_idx = torch.topk(entropy_above_threshold, M)
            idx = topk_idx
        else:
            raise ValueError(f"不支持的采样模式: {mode}")
        # 获取采样坐标和标签
        sampled_coords = coords_all[idx]  # (M, spatial_dim)
        sampled_labels = torch.full((M,), label_value, dtype=torch.long, device=prob_map.device)
        return sampled_coords, sampled_labels

    def _extract_point_prompts(
        self,
        logits: torch.Tensor,
        num_points_per_slice: int,  # 参数更新
        threshold: float,
        sample_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C = logits.shape[:2]         # B: batch size, C: 类别数
        spatial_dims = logits.shape[2:]  # 获取空间维度HWD
        spatial_dim_count = len(spatial_dims)  # 空间维度的数量 (例如 3D 为 3)
        device = logits.device

        # 检查维度
        if spatial_dim_count != 3:
            raise ValueError(
                f"期望 3D 空间维度 (例如 HWD)，但得到 {spatial_dim_count} 维: {spatial_dims}")

        num_slices = spatial_dims[-1]  # 层数

        if self.debug:
            print(f'---------------------Debug信息----------------------------')
            print(f"输入 logits 形状: {logits.shape}")
            print(f"空间维度: {spatial_dims}, 切片数 (假设是最后一维): {num_slices}")
            print(f"每个类别每片采样点数: {num_points_per_slice}")
            print(f'---------------------Debug信息----------------------------')

        # 计算概率图
        probs = logits # 已经获得了对应的概率图

        batch_coords: List[torch.Tensor] = []  # 存储每个 batch 样本的坐标
        batch_labels: List[torch.Tensor] = []  # 存储每个 batch 样本的标签

        # 遍历 batch 中的每个样本
        for b in range(B):
            coords_per_sample: List[torch.Tensor] = []  # 当前样本收集的所有坐标
            labels_per_sample: List[torch.Tensor] = []  # 当前样本收集的所有标签

            # 遍历每个前景类别
            for c in range(1, C):
                if self.debug:
                    print(f'\n[样本 {b+1}/{B}, 类别 {c}] 处理中...')

                # 当前类别当前样本的所有切片坐标
                coords_per_class_sample: List[torch.Tensor] = []
                # 当前类别当前样本的所有切片标签
                labels_per_class_sample: List[torch.Tensor] = []

                # 获取当前类别 c 的概率图，形状HWD
                prob_map_3d = probs[b, c]

                # 遍历每个切片
                for z in range(num_slices):
                    # 获取当前 2D 切片 (形状: [D, H])
                    prob_map_slice = prob_map_3d[..., z]  # 取最后一个维度

                    # 在当前 2D 切片上采样点
                    coords_2d, labels_slice = self._sample_points_from_mask(
                        prob_map=prob_map_slice,
                        label_value=c,
                        num_points=num_points_per_slice,  # 使用每片采样数
                        threshold=threshold,
                        mode=sample_mode,
                    )

                    # 如果在当前切片上采样到了点，那么把2D坐标恢复成3D坐标
                    if coords_2d.numel() > 0:
                        z_coords = torch.full_like(
                            labels_slice, z, dtype=torch.long).unsqueeze(1)
                        coords_3d = torch.cat((coords_2d, z_coords), dim=1)

                        # 收集当前切片的 3D 坐标和标签
                        coords_per_class_sample.append(coords_3d)
                        labels_per_class_sample.append(labels_slice)

                # 如果当前类别在该样本的所有切片中采样到了点
                if coords_per_class_sample:
                    # 将该类别的所有点连接起来
                    coords_cat_class = torch.cat(
                        coords_per_class_sample, dim=0)
                    labels_cat_class = torch.cat(
                        labels_per_class_sample, dim=0)
                    if self.debug:
                        print(
                            f'[样本 {b+1}/{B}, 类别 {c}] 总共采样到 {coords_cat_class.shape[0]} 个点')
                    # 添加到当前样本的总列表中
                    coords_per_sample.append(coords_cat_class)
                    labels_per_sample.append(labels_cat_class)
                elif self.debug:
                    print(f'[样本 {b+1}/{B}, 类别 {c}] 未采样到任何点')

            if self.debug:
                print(f'\n[样本 {b+1}/{B}, 背景类别 0] 处理中...')
            coords_bg_sample: List[torch.Tensor] = []
            labels_bg_sample: List[torch.Tensor] = []
            prob_map_bg_3d = probs[b, 0]  # 背景概率图

            for z in range(num_slices):
                prob_map_bg_slice = prob_map_bg_3d[..., z]
                coords_bg_2d, labels_bg_slice = self._sample_points_from_mask(
                    prob_map=prob_map_bg_slice,
                    label_value=0,  # 背景标签为 0
                    num_points=num_points_per_slice,
                    threshold=threshold,
                    mode=sample_mode,
                )
                if coords_bg_2d.numel() > 0:
                    z_coords = torch.full_like(
                        labels_bg_slice, z, dtype=torch.long).unsqueeze(1)
                    coords_bg_3d = torch.cat((coords_bg_2d, z_coords), dim=1)
                    coords_bg_sample.append(coords_bg_3d)
                    labels_bg_sample.append(labels_bg_slice)
                    # if self.debug:
                    #     print(f'  切片 {z}: 采样到 {coords_bg_3d.shape[0]} 个背景点')

            if coords_bg_sample:
                coords_cat_bg = torch.cat(coords_bg_sample, dim=0)
                labels_cat_bg = torch.cat(labels_bg_sample, dim=0)
                if self.debug:
                    print(
                        f'[样本 {b+1}/{B}, 背景类别 0] 总共采样到 {coords_cat_bg.shape[0]} 个点')
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
                    torch.zeros(1, spatial_dim_count,
                                dtype=torch.long, device=device)
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

        coords_padded = pad_sequence(
            batch_coords, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(
            batch_labels, batch_first=True, padding_value=-1)  # 使用 -1 标记填充标签

        return coords_padded, labels_padded

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 检查输入维度是否正确
        if x.dim() < 5:
            raise ValueError(f"输入张量应至少为 5D (B, C, spatial...), 得到 {x.shape}")

        # 获取配置参数
        n_pts_slice = self.num_points_per_slice # 每个切片的点数量
        thr = self.threshold # 设置概率阈值
        mode = self.sample_mode # 采样模式

        # 使用不确定性分析获得粗分割概率图
        logits = self.bayesian_generator(x)

        # 从概率图提取提取点 prompts
        coords, labels = self._extract_point_prompts(
            logits, n_pts_slice, thr, mode)

        return coords, labels

def PromptGenerator_test_v2():
    input_shape = (1, 1, 112, 112, 80)  # BCHWD
    n_classes = 2                        # 2个类别: 0=背景, 1=前景
    num_points_per_slice = 5            # 每个类别在每个 Z 切片上采样 10 个点
    threshold = 0.5                      # 概率阈值
    sample_mode = 'entropy'                 # 采样模式 ('random' 或 'topk')
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

    with torch.no_grad():  # 在评估时不计算梯度
        coords_output, labels_output = model(mock_input)

    # --- 打印输出结果 ---
    print("\n--- 测试结果 ---")
    print(f"输出坐标形状 (coords_output.shape): {coords_output.shape}")
    print(f"输出标签形状 (labels_output.shape): {labels_output.shape}")

    # 分析预期点数
    num_slices = input_shape[-1]  # D = 80
    max_expected_points_per_sample = n_classes * num_slices * num_points_per_slice
    print(f"\n理论上每个样本最多采样点数 (如果所有切片所有类别都满足条件):")
    print(f"  {n_classes} (类别数) * {num_slices} (切片数) * {num_points_per_slice} (每片点数) = {max_expected_points_per_sample}")
    print(f"实际采样点数 (输出张量的第二维): {coords_output.shape[1]}")
    print(
        f"(注意：实际点数会少于最大值，因为某些切片/类别的概率可能不满足阈值，或者候选点不足 {num_points_per_slice} 个)")

    # 打印第一个样本的部分采样点和标签
    if coords_output.shape[1] > 0:  # 确保有采样点
        print("\n第一个样本的部分输出坐标 (前 5 个点):")
        print(coords_output[0, :5, :])
        print("\n第一个样本的部分输出标签 (前 5 个点):")
        print(labels_output[0, :5])

        # 检查坐标范围是否合理
        spatial_dims_shape = input_shape[2:]  # (112, 112, 80)
        coords_np = coords_output[0].cpu().numpy()  # 转到 CPU 并转为 NumPy
        labels_np = labels_output[0].cpu().numpy()
        valid_mask = labels_np != -1  # 排除填充点
        valid_coords = coords_np[valid_mask]

        if valid_coords.shape[0] > 0:
            min_coords = valid_coords.min(axis=0)
            max_coords = valid_coords.max(axis=0)
            print("\n第一个样本有效采样点坐标范围:")
            print(
                f"  维度 H (期望 [0, {spatial_dims_shape[0]-1}]): Min={min_coords[0]}, Max={max_coords[0]}")
            print(
                f"  维度 W (期望 [0, {spatial_dims_shape[1]-1}]): Min={min_coords[1]}, Max={max_coords[1]}")
            print(
                f"  维度 D (期望 [0, {spatial_dims_shape[2]-1}]): Min={min_coords[2]}, Max={max_coords[2]}")
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
            dim=192,  # 第一个block的维度为192
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=8,  # 如果不设置window_size会进行全局注意力，导致显存爆炸
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
            window_size=8,  # 如果不设置window_size会进行全局注意力，导致显存爆炸
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
        x = self.block3(x)
        x = x.permute(0, 2, 3, 1, 4)

        return x

def ImageEncoderViT3D_test():
    # 实例化模型
    model = ImageEncoderViT3D(img_size=(112, 112, 80),
                              patch_size=8,
                              embed_dim=192,
                              num_heads=12,
                              mlp_ratio=4.0,
                              qkv_bias=True,
                              norm_layer=nn.LayerNorm,
                              act_layer=nn.GELU,
                              use_abs_pos=True,
                              use_rel_pos=False,
                              rel_pos_zero_init=True,
                              window_size=0,
                              global_attn_indexes=(),
                              )
    model.eval()  # 设置为评估模式

    # 创建模拟输入张量 (B, C, D, H, W)
    # B=1, C=1, D=112, H=112, W=80
    mock_input_tensor = torch.randn(1, 1, 112, 112, 80)
    print(f"输入张量形状: {mock_input_tensor.shape}")

    # 通过模型进行前向传播
    with torch.no_grad():  # 在测试时不需要计算梯度
        output_tensor = model(mock_input_tensor)

    # 打印输出张量的形状
    print(f"输出张量形状: {output_tensor.shape}")
    summary(model=model, input_size=(1, 1, 112, 112, 80))

class MaskDecoder3D(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        # transformer: nn.Module ,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:

        super().__init__()
        self.transformer_dim = transformer_dim
        # self.transformer = transformer
        self.transformer = TwoWayTransformer3D(
            depth=2,
            embedding_dim=self.transformer_dim,
            mlp_dim=2048,
            num_heads=8,
        )

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose3d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm3d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose3d(transformer_dim // 4,
                               transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 输入张量
        print(f"image_embeddings shape: {image_embeddings.shape}")
        print(f"image_pe shape: {image_pe.shape}")
        print(
            f"sparse_prompt_embeddings shape: {sparse_prompt_embeddings.shape}")
        print(
            f"dense_prompt_embeddings shape: {dense_prompt_embeddings.shape}")

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(
                image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        if image_pe.shape[0] != tokens.shape[0]:
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        else:
            pos_src = image_pe
        b, c, x, y, z = src.shape

        # Run the transformer
        # import IPython; IPython.embed()
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, x, y, z)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, x, y, z = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b,
                 c, x * y * z)).view(b, -1, x, y, z)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

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
            num_points_per_slice: int = 5,  # 每个切片的Prompt的数量
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
        self.num_points_per_slice = num_points_per_slice  # 添加切片参数
        self.threshold = threshold
        self.generatorways = generatorways
        self.debug = debug

        # ------- PromptEncoder参数 -------
        self.embedding_size = tuple(
            s // patch_size for s in image_size)  # e.g., 112 -> 14
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
            num_points_per_slice=self.num_points_per_slice,  # 添加切片中prompt数量
            threshold=self.threshold,
            sample_mode=self.generatorways,
            debug=self.debug
        )

        # ------- Prompt Encoder -------
        self.promptencoder = PromptEncoder3D(
            embed_dim=self.embed_dim,                     # e.g., 768
            image_embedding_size=self.embedding_size,     # e.g., (14, 14, 10)
            # e.g., (112, 112, 80)
            input_image_size=self.image_size,
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
            masks = F.interpolate(masks, size=original_size,
                                  mode="trilinear", align_corners=False)

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
            points=None,  # 这里不设置prompt
            boxes=None,
            masks=None
        )

        print(f'sparse_embeddings shape: {sparse_embeddings.shape}')
        print(f'dense_embeddings shape: {dense_embeddings.shape}')

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
        sam_output = self.postprocess_masks(
            after_maskencoder, input_size=x.shape[-3:], original_size=x.shape[-3:])

        # 测试过程中设置模拟输出
        vnet_output = torch.randn(1, 2, 112, 112, 80)
        sam_output = torch.randn(1, 2, 112, 112, 80)

        return vnet_output, sam_output

def networktest():
    # 检查CUDA是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU")

    # 实例化网络
    model = Network(in_channels=1,
                    encoder_depth=4,
                    num_points_per_slice=5,
                    embed_dim=384,
                    out_chans=768).to(device=device)

    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device=device)
    vnet_output, sam_output = model(input_tensor)

    # summary(model=model,input_size=(1,1,112,112,80))

    return

if __name__ == "__main__":
    # networktest()
    # ImageEncoderViT3D_test()
    # PromptGenerator_test_v2()
    # test_BayesianProbabilityGenerator()