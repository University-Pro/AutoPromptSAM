# 用于测试通过解耦不确定性实现更加优质的点prompt
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

# 导入绘图的相关模块
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
import seaborn as sns

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
from networks.VNet_MultiOutput import VNet

# 创建有明显前景背景的模拟张量
def create_complex_3d_tensor(
    tensor_size=(1, 1, 112, 112, 80),
    device='cuda',
    center=(56, 56, 40),       # 形状中心坐标
    shape_type='ellipsoid',      # 支持的形状类型：'sphere', 'ellipsoid', 'ring', 'cylinder', 'pyramid'
    radius=15,                # 基本半径/尺寸
    noise_level=0.1,           # 背景噪声强度，范围在 [0, 1]
    foreground_brightness=1.0, # 前景亮度，范围在 [0, 1]
    ellipsoid_axes=(15, 15, 15),   # 椭球体的三个轴长度（仅当 shape_type='ellipsoid' 时有效）
    ring_inner_radius=8,      # 环状结构内部半径（仅当 shape_type='ring' 时有效）
    cylinder_height=30,       # 圆柱高度（仅当 shape_type='cylinder' 时有效）
    pyramid_base_size=(20, 20) # 金字塔底面尺寸（仅当 shape_type='pyramid' 时有效）
):
    """
    创建一个包含复杂三维形状的张量，并添加背景噪声。
    Args:
        tensor_size (tuple): 张量维度 (batch_size, channels, height, width, depth)
        device (str): 设备类型（'cuda' 或 'cpu'）
        center (tuple): 形状中心坐标 (x, y, z)
        shape_type (str): 支持的形状类型：'sphere', 'ellipsoid', 'ring', 'cylinder', 'pyramid'
        radius (int): 基本半径/尺寸
        noise_level (float): 背景噪声强度，范围在 [0, 1]
        foreground_brightness (float): 前景亮度，范围在 [0, 1]
        ellipsoid_axes (tuple): 椭球体的三个轴长度（仅当 shape_type='ellipsoid' 时有效）
        ring_inner_radius (int): 环状结构内部半径（仅当 shape_type='ring' 时有效）
        cylinder_height (int): 圆柱高度（仅当 shape_type='cylinder' 时有效）
        pyramid_base_size (tuple): 金字塔底面尺寸（仅当 shape_type='pyramid' 时有效）
    Returns:
        torch.Tensor: 包含复杂三维形状和背景噪声的张量
    """
    # 创建全零的背景张量
    dummy_input = torch.zeros(tensor_size).to(device)
    
    # 中心坐标
    center_x, center_y, center_z = center
    
    # 填充形状区域
    for x in range(center_x - radius, center_x + radius):
        for y in range(center_y - radius, center_y + radius):
            for z in range(center_z - radius, center_z + radius):
                # 计算相对坐标
                dx = x - center_x
                dy = y - center_y
                dz = z - center_z
                
                if shape_type == 'sphere':
                    # 球体条件判断
                    distance_sq = dx**2 + dy**2 + dz**2
                    condition = (distance_sq <= radius**2)
                    
                elif shape_type == 'ellipsoid':
                    # 椭球体条件判断，使用不同的轴长
                    a, b, c = ellipsoid_axes
                    normalized_x = (dx / a)**2
                    normalized_y = (dy / b)**2
                    normalized_z = (dz / c)**2
                    condition = (normalized_x + normalized_y + normalized_z <= 1)
                    
                elif shape_type == 'ring':
                    # 环状结构：在球体内挖去一个较小的同心球体
                    distance_sq = dx**2 + dy**2 + dz**2
                    inner_condition = (distance_sq > ring_inner_radius**2)
                    outer_condition = (distance_sq <= radius**2)
                    condition = (inner_condition & outer_condition)
                    
                elif shape_type == 'cylinder':
                    # 圆柱体，沿着z轴延伸
                    distance_sq_xy = dx**2 + dy**2
                    cylinder_condition = (distance_sq_xy <= radius**2)
                    height_condition = (abs(dz) <= cylinder_height // 2)
                    condition = (cylinder_condition & height_condition)
                    
                elif shape_type == 'pyramid':
                    # 简单金字塔形状，从中心向外扩展
                    base_x, base_y = pyramid_base_size
                    slope_z = abs(dz) * (radius / (base_x // 2))
                    condition = (
                        (abs(dx) <= radius) & 
                        (abs(dy) <= radius) & 
                        (abs(dx) + abs(dy) >= slope_z)
                    )
                
                else:
                    raise ValueError(f"不支持的形状类型：{shape_type}")
                
                if condition:
                    dummy_input[0, 0, x, y, z] = foreground_brightness
    
    # 添加背景噪声
    noise = noise_level * torch.randn_like(dummy_input)
    dummy_input += noise
    
    # 将数据限制在 [0, 1] 范围内
    dummy_input = torch.clamp(dummy_input, 0.0, 1.0)
    
    return dummy_input

def visualize_3d_prompts_and_tensors(class_prompts, tensors_dict, save_path=None, 
                                   slice_step=5, max_points_per_class=10):
    """
    创建3D可视化，显示选择的提示点和对应的张量
    
    参数:
        class_prompts: 字典，包含每个类别的提示点
        tensors_dict: 字典，包含要可视化的张量数据
            - 'input': 输入张量 [D, H, W]
            - 'epistemic': 模型不确定性 [D, H, W] 
            - 'aleatoric': 数据不确定性 [D, H, W]
            - 'score': 提示点分数 [D, H, W]
        save_path: 保存路径
        slice_step: 切片间隔
        max_points_per_class: 每个类别显示的最大点数
    """
    
    # 设置图形
    fig = plt.figure(figsize=(20, 16))
    
    # 颜色映射
    colors_class = ['red', 'blue', 'green', 'orange', 'purple']
    
    # 1. 输入数据的3D可视化
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    input_tensor = tensors_dict['input']
    D, H, W = input_tensor.shape
    
    # 创建体积渲染效果
    threshold = 0.3  # 只显示高于阈值的体素
    mask = input_tensor > threshold
    z_coords, y_coords, x_coords = np.where(mask.cpu().numpy())
    colors = input_tensor[mask].cpu().numpy()
    
    scatter1 = ax1.scatter(x_coords, y_coords, z_coords, 
                          c=colors, cmap='viridis', alpha=0.6, s=20)
    ax1.set_title('Input Data (3D Volume)')  # 改为英文
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5)
    
    # 2. 模型不确定性可视化
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    epistemic_tensor = tensors_dict['epistemic']
    
    # 只显示高不确定性的区域
    high_uncertainty = epistemic_tensor > np.percentile(epistemic_tensor.cpu().numpy(), 80)
    z_coords, y_coords, x_coords = np.where(high_uncertainty.cpu().numpy())
    colors = epistemic_tensor[high_uncertainty].cpu().numpy()
    
    scatter2 = ax2.scatter(x_coords, y_coords, z_coords,
                          c=colors, cmap='Reds', alpha=0.7, s=25)
    ax2.set_title('Model Uncertainty (High Regions)') 
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5)
    
    # 3. 数据不确定性可视化
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    aleatoric_tensor = tensors_dict['aleatoric']
    
    high_aleatoric = aleatoric_tensor > np.percentile(aleatoric_tensor.cpu().numpy(), 80)
    z_coords, y_coords, x_coords = np.where(high_aleatoric.cpu().numpy())
    colors = aleatoric_tensor[high_aleatoric].cpu().numpy()
    
    scatter3 = ax3.scatter(x_coords, y_coords, z_coords,
                          c=colors, cmap='Blues', alpha=0.7, s=25)
    ax3.set_title('Data Uncertainty (High Regions)')
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    plt.colorbar(scatter3, ax=ax3, shrink=0.5)
    
    # 4. 提示点分数可视化
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    score_tensor = tensors_dict['score']
    
    high_score = score_tensor > np.percentile(score_tensor.cpu().numpy(), 85)
    z_coords, y_coords, x_coords = np.where(high_score.cpu().numpy())
    colors = score_tensor[high_score].cpu().numpy()
    
    scatter4 = ax4.scatter(x_coords, y_coords, z_coords,
                          c=colors, cmap='plasma', alpha=0.7, s=25)
    ax4.set_title('Prompt Suitability Score (High Regions)')
    ax4.set_xlabel('X'); ax4.set_ylabel('Y'); ax4.set_zlabel('Z')
    plt.colorbar(scatter4, ax=ax4, shrink=0.5)
    
    # 5. 选中的提示点可视化
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    
    # 显示背景体积（半透明）
    bg_mask = input_tensor > threshold
    z_bg, y_bg, x_bg = np.where(bg_mask.cpu().numpy())
    ax5.scatter(x_bg, y_bg, z_bg, c='lightgray', alpha=0.1, s=5)
    
    # 显示每个类别的提示点
    legend_elements = []
    for class_id, prompts in class_prompts.items():
        if not prompts:
            continue
            
        # 限制显示的点数
        prompts_to_show = prompts[:max_points_per_class]
        
        coords = [[p['coords_3d'][1], p['coords_3d'][0], p['coords_3d'][2]] for p in prompts_to_show]  # [x, y, z]
        coords = np.array(coords)
        
        if len(coords) > 0:
            color = colors_class[class_id % len(colors_class)]
            ax5.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                       c=color, s=100, alpha=0.9, marker='o', edgecolors='black', linewidth=2)
            
            # 添加点的编号
            for i, (x, y, z) in enumerate(coords):
                ax5.text(x, y, z, f'{class_id}-{i+1}', fontsize=8)
            
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=10,
                                    label=f'Class {class_id} ({len(prompts_to_show)} points)'))
    
    ax5.set_title('Selected Prompt Points')
    ax5.set_xlabel('X'); ax5.set_ylabel('Y'); ax5.set_zlabel('Z')
    ax5.legend(handles=legend_elements, loc='upper right')
    
    # 6. 综合视图（切片+点）
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    
    # 显示几个关键切片
    for z_slice in range(0, D, slice_step):
        if z_slice >= D:
            break
        
        slice_data = input_tensor[z_slice].cpu().numpy()
        y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        # 只显示有数据的区域
        mask_2d = slice_data > threshold
        if mask_2d.sum() > 0:
            ax6.scatter(x_grid[mask_2d], y_grid[mask_2d], 
                       np.full(mask_2d.sum(), z_slice),
                       c=slice_data[mask_2d], cmap='viridis', 
                       alpha=0.3, s=10)
    
    # 叠加提示点
    for class_id, prompts in class_prompts.items():
        if not prompts:
            continue
            
        prompts_to_show = prompts[:max_points_per_class]
        coords = [[p['coords_3d'][1], p['coords_3d'][0], p['coords_3d'][2]] for p in prompts_to_show]
        coords = np.array(coords)
        
        if len(coords) > 0:
            color = colors_class[class_id % len(colors_class)]
            ax6.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                       c=color, s=150, alpha=1.0, marker='*', 
                       edgecolors='white', linewidth=2)
    
    ax6.set_title('Composite View (Slices + Points)')
    ax6.set_xlabel('X'); ax6.set_ylabel('Y'); ax6.set_zlabel('Z')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D可视化已保存到: {save_path}")
    
    return fig

def create_detailed_point_analysis(class_prompts, tensors_dict, top_n=5):
    """
    创建详细的提示点分析图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 收集所有点的数据
    all_points_data = []
    for class_id, prompts in class_prompts.items():
        for prompt in prompts:
            y, x, z = prompt['coords_3d']
            all_points_data.append({
                'class_id': class_id,
                'coords': [x, y, z],
                'confidence': prompt['class_probability'],
                'score': prompt['suitability_score'],
                'epistemic': tensors_dict['epistemic'][z, y, x].item(),
                'aleatoric': tensors_dict['aleatoric'][z, y, x].item()
            })
    
    # 1. 置信度分布
    ax1 = axes[0, 0]
    class_confidences = {}
    for point in all_points_data:
        class_id = point['class_id']
        if class_id not in class_confidences:
            class_confidences[class_id] = []
        class_confidences[class_id].append(point['confidence'])
    
    for class_id, confidences in class_confidences.items():
        ax1.hist(confidences, alpha=0.7, label=f'类别 {class_id}', bins=20)
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Prompt Point Confidence Distribution')
    ax1.legend()
    
    # 2. 不确定性散点图
    ax2 = axes[0, 1]
    for class_id in class_prompts.keys():
        class_points = [p for p in all_points_data if p['class_id'] == class_id]
        if class_points:
            epistemic_vals = [p['epistemic'] for p in class_points]
            aleatoric_vals = [p['aleatoric'] for p in class_points]
            ax2.scatter(epistemic_vals, aleatoric_vals, 
                       label=f'类别 {class_id}', alpha=0.7, s=50)
    
    ax2.set_xlabel('Model Uncertainty')
    ax2.set_ylabel('Data Uncertainty')  
    ax2.set_title('Prompt Point Uncertainty Distribution')
    ax2.legend()
    
    # 3. 分数排名
    ax3 = axes[1, 0]
    sorted_points = sorted(all_points_data, key=lambda x: x['score'], reverse=True)
    top_points = sorted_points[:top_n*2]  # 每类取top_n个
    
    scores = [p['score'] for p in top_points]
    colors = [f'C{p["class_id"]}' for p in top_points]
    bars = ax3.bar(range(len(scores)), scores, color=colors, alpha=0.7)
    
    ax3.set_xlabel('Point Rank')
    ax3.set_ylabel('Suitability Score')
    ax3.set_title(f'Top-{len(top_points)} Prompt Points')
    
    # 添加类别标签
    for i, (bar, point) in enumerate(zip(bars, top_points)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'C{point["class_id"]}', ha='center', va='bottom', fontsize=8)
    
    # 4. 空间分布热图（2D投影）
    ax4 = axes[1, 1]
    H, W = tensors_dict['input'].shape[1], tensors_dict['input'].shape[2]
    
    # 创建2D密度图
    density_map = np.zeros((H, W))
    for point in all_points_data:
        x, y, z = point['coords']
        if 0 <= y < H and 0 <= x < W:
            density_map[y, x] += 1
    
    im = ax4.imshow(density_map, cmap='hot', interpolation='nearest')
    ax4.set_title('Prompt Points Spatial Distribution (2D Projection)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    plt.colorbar(im, ax=ax4)
    
    # 叠加点位置
    for class_id in class_prompts.keys():
        class_points = [p for p in all_points_data if p['class_id'] == class_id]
        if class_points:
            xs = [p['coords'][0] for p in class_points]
            ys = [p['coords'][1] for p in class_points]
            ax4.scatter(xs, ys, c=f'C{class_id}', s=30, alpha=0.8, 
                       edgecolors='white', linewidth=1)
    
    plt.tight_layout()
    return fig, all_points_data

def set_mc_dropout_mode(model: nn.Module):
    """
    正确的MC Dropout模式设置：
    开启Dropout，但将所有Normalization层（BatchNorm, InstanceNorm等）固定在评估模式。
    这是获取稳定且有意义不确定性的关键。
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
        elif m.__class__.__name__.startswith(('BatchNorm', 'InstanceNorm')):
            m.eval()

class DecoupledUncertaintyGenerator(nn.Module):
    def __init__(self, network: nn.Module, num_mc_samples: int = 30):
        super().__init__()
        if num_mc_samples <= 1:
            raise ValueError("num_mc_samples must be > 1.")
        
        self.network = network
        self.num_mc_samples = num_mc_samples
        
        set_mc_dropout_mode(self.network) # 确保dropout在评估时也开启

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        执行MC采样并返回解耦的 H_epistemic 和 H_aleatoric 地图。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - H_epistemic (torch.Tensor): 模型不确定性地图, Shape: [B, 1, D, H, W]
            - H_aleatoric (torch.Tensor): 数据不确定性地图, Shape: [B, 1, D, H, W]
        """
        all_softmax_outputs = []
        all_aleatoric_vars = []

        for _ in range(self.num_mc_samples):
            # 假设你的VNetAleatoric返回两个值
            logits, log_aleatoric_var = self.network(x)
            
            # 1. 收集softmax输出，用于计算模型不确定性
            softmax_out = F.softmax(logits, dim=1)
            all_softmax_outputs.append(softmax_out)
            
            # 2. 收集数据不确定性
            aleatoric_var = torch.exp(log_aleatoric_var)
            all_aleatoric_vars.append(aleatoric_var)

        # Shape: [num_samples, B, C, D, H, W]
        stacked_softmax = torch.stack(all_softmax_outputs)
        stacked_aleatoric = torch.stack(all_aleatoric_vars)

        # --- 计算 H_epistemic (模型不确定性) ---
        # H_epistemic 可以通过预测概率的方差来近似
        # 这是衡量模型在不同MC样本间分歧的直接方式
        variance_of_probs = torch.var(stacked_softmax, dim=0) # Shape: [B, C, D, H, W]
        # 我们将所有类别的方差相加，得到每个像素的一个总不确定性分数
        H_epistemic = torch.sum(variance_of_probs, dim=1, keepdim=True) # Shape: [B, 1, D, H, W]

        # --- 计算 H_aleatoric (数据不确定性) ---
        # H_aleatoric 是网络对数据噪声的直接预测，我们取其在MC样本间的期望
        mean_aleatoric_var = torch.mean(stacked_aleatoric, dim=0) # Shape: [B, C, D, H, W]
        # 同样，将所有类别的方差相加
        H_aleatoric = torch.sum(mean_aleatoric_var, dim=1, keepdim=True) # Shape: [B, 1, D, H, W]

        # 你也可以返回平均概率用于最终分割
        mean_probs = torch.mean(stacked_softmax, dim=0)

        return H_epistemic, H_aleatoric, mean_probs

def find_top_k_prompts_per_class(score_map: torch.Tensor, segmentation_probs: torch.Tensor, 
                                k_per_class: int = 20, num_classes: int = 2, 
                                suppression_radius: int = 5) -> dict:
    """
    为每个类别通过非极大值抑制迭代寻找最优K个提示点
    
    参数：
        score_map: 3D张量[D,H,W]，表示提示点适宜性得分
        segmentation_probs: 4D张量[C,H,W,D]，每个位置的类别概率
        k_per_class: 每个类别需要的提示点数量
        num_classes: 类别总数
        suppression_radius: 抑制半径（像素单位）
        
    返回：
        字典，键为类别ID，值为该类别的提示点列表
    """
    d, h, w = score_map.shape
    c, h_seg, w_seg, d_seg = segmentation_probs.shape
    
    # 确保维度匹配
    assert h == h_seg and w == w_seg and d == d_seg, "score_map和segmentation_probs维度不匹配"
    
    # 预建坐标网格提升效率
    zz, yy, xx = torch.meshgrid(
        torch.arange(d, device=score_map.device),
        torch.arange(h, device=score_map.device),
        torch.arange(w, device=score_map.device),
        indexing='ij'
    )
    
    class_prompts = {}
    
    for class_id in range(num_classes):
        class_prompts[class_id] = []
        
        # 为当前类别创建掩码：只考虑该类别概率最高的像素
        class_mask = torch.argmax(segmentation_probs, dim=0) == class_id  # Shape: [H, W, D]
        class_mask = class_mask.permute(2, 0, 1)  # 转换为 [D, H, W]
        
        # 创建当前类别的分数地图
        class_score_map = score_map.clone()
        class_score_map[~class_mask] = -float('inf')  # 将不属于当前类别的像素设为负无穷
        
        temp_score_map = class_score_map.clone()
        
        print(f"\n开始为类别 {class_id} 寻找 {k_per_class} 个提示点...")
        
        for i in range(k_per_class):
            best_idx = torch.argmax(temp_score_map)
            best_score = temp_score_map.reshape(-1)[best_idx]
            
            if best_score == -float('inf'):
                print(f"类别 {class_id} 只能找到 {i} 个有效点")
                break  # 无有效点时提前终止
            
            # 转换坐标
            z = best_idx // (h * w)
            y = (best_idx % (h * w)) // w
            x = best_idx % w
            
            # 获取该点的详细信息
            coords_3d = [y.item(), x.item(), z.item()]  # [H, W, D] 格式
            
            # 获取该点的类别概率
            point_probs = segmentation_probs[:, y, x, z]  # Shape: [C]
            predicted_class = torch.argmax(point_probs).item()
            class_probability = torch.max(point_probs).item()
            
            # 确认该点确实属于当前类别
            if predicted_class != class_id:
                print(f"警告: 点 [{y}, {x}, {z}] 预测类别 {predicted_class} 与目标类别 {class_id} 不匹配")
            
            # 存储结果
            prompt_info = {
                'rank': i + 1,
                'coords_3d': coords_3d,
                'predicted_class': predicted_class,
                'target_class': class_id,
                'class_probability': class_probability,
                'suitability_score': best_score.item(),
                'all_class_probs': point_probs.tolist()
            }
            
            class_prompts[class_id].append(prompt_info)
            
            # 球形抑制区域
            suppression_mask = ((zz-z)**2 + (yy-y)**2 + (xx-x)**2) < suppression_radius**2
            temp_score_map[suppression_mask] = -float('inf')
        
        print(f"类别 {class_id} 成功找到 {len(class_prompts[class_id])} 个提示点")
    
    return class_prompts

# 修改主测试函数，添加可视化调用
def DecoupledUncertaintyGenerator_test_with_visualization():
    """
    带3D可视化的不确定性生成器测试
    """
    # --- 原有的测试代码 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练网络
    trained_vnet_aleatoric = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True)
    trained_vnet_aleatoric.load_state_dict(torch.load('result/VNet_Multi/LA_16/Pth/Best.pth', map_location=device))
    trained_vnet_aleatoric.to(device)

    # --- 2. 实例化新的不确定性生成器 ---
    uncertainty_gen = DecoupledUncertaintyGenerator(network=trained_vnet_aleatoric, num_mc_samples=25)
    
    # 创建模拟的输入数据
    dummy_input = create_complex_3d_tensor(
        tensor_size=(1, 1, 112, 112, 80),  # [B, C, D, H, W]
        device=device,
        center=(56, 56, 40),       # 中心坐标
        shape_type='ellipsoid',      # 支持的形状类型：'sphere', 'ellipsoid', 'ring', 'cylinder', 'pyramid'
        radius=15,                # 基本半径/尺寸
        noise_level=0.1,           # 背景噪声强度，范围在 [0, 1]
        foreground_brightness=1.0, # 前景亮度，范围在 [0, 1]
        ellipsoid_axes=(15, 15, 15)   # 椭球体的三个轴长度（仅当 shape_type='ellipsoid' 时有效）
    )
    
    H_epistemic, H_aleatoric, final_segmentation_probs = uncertainty_gen(dummy_input)
    
    print(f"模型不确定性 H_epistemic shape: {H_epistemic.shape}")
    print(f"数据不确定性 H_aleatoric shape: {H_aleatoric.shape}")
    print(f'最终分割概率图 final_segmentation_probs shape: {final_segmentation_probs.shape}')
    
    # 计算提示点适宜性分数
    def normalize(tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    
    H_epistemic_norm = normalize(H_epistemic)
    H_aleatoric_norm = normalize(H_aleatoric)
    
    w_epistemic = 0.5
    w_aleatoric = 0.5
    prompt_suitability_score = w_epistemic * (1 - H_epistemic_norm) + w_aleatoric * (1 - H_aleatoric_norm)
    
    # 转换格式用于NMS
    score_map = prompt_suitability_score.squeeze(0).squeeze(0).permute(2, 0, 1)  # [D, H, W]
    segmentation_probs = final_segmentation_probs.squeeze(0).permute(0, 1, 2, 3)  # [C, H, W, D]
    
    # 使用您的find_top_k_prompts_per_class函数（这里简化实现）
    class_prompts = simulate_class_prompts(score_map, segmentation_probs, k_per_class=20)
    
    # 准备可视化数据
    tensors_dict = {
        'input': dummy_input.squeeze(0).squeeze(0).permute(2, 0, 1),  # [D, H, W]
        'epistemic': H_epistemic.squeeze(0).squeeze(0).permute(2, 0, 1),  # [D, H, W]
        'aleatoric': H_aleatoric.squeeze(0).squeeze(0).permute(2, 0, 1),  # [D, H, W]
        'score': score_map  # [D, H, W]
    }
    
    # 创建3D可视化
    print("\n正在生成3D可视化...")
    fig1 = visualize_3d_prompts_and_tensors(
        class_prompts=class_prompts,
        tensors_dict=tensors_dict,
        save_path='3d_prompts_visualization.png',
        max_points_per_class=10
    )
    
    # 创建详细分析图
    print("正在生成详细分析图...")
    fig2, point_data = create_detailed_point_analysis(
        class_prompts=class_prompts,
        tensors_dict=tensors_dict,
        top_n=10
    )
    
    # 输出统计信息
    print(f"\n=== 可视化统计信息 ===")
    total_points = sum(len(prompts) for prompts in class_prompts.values())
    print(f"总共可视化了 {total_points} 个提示点")
    
    for class_id, prompts in class_prompts.items():
        print(f"类别 {class_id}: {len(prompts)} 个点")
    
    return class_prompts, tensors_dict, fig1, fig2

def simulate_class_prompts(score_map, segmentation_probs, k_per_class=20):
    """
    模拟类别提示点生成（用于演示可视化）
    在实际使用中，请替换为您的find_top_k_prompts_per_class函数
    """
    import random
    
    D, H, W = score_map.shape
    C = segmentation_probs.shape[0]
    
    class_prompts = {}
    
    for class_id in range(C):
        class_prompts[class_id] = []
        
        # 找到该类别概率较高的区域
        pred_mask = torch.argmax(segmentation_probs, dim=0) == class_id
        valid_indices = torch.where(pred_mask)
        
        if len(valid_indices[0]) == 0:
            continue
            
        # 随机选择一些点（在实际实现中应该基于分数）
        num_valid = len(valid_indices[0])
        selected_indices = random.sample(range(num_valid), min(k_per_class, num_valid))
        
        for i, idx in enumerate(selected_indices):
            y = valid_indices[0][idx].item()
            x = valid_indices[1][idx].item()
            z = valid_indices[2][idx].item()
            
            coords_3d = [y, x, z]  # [H, W, D] 格式
            
            # 获取该点的概率和分数
            point_probs = segmentation_probs[:, y, x, z]
            predicted_class = torch.argmax(point_probs).item()
            class_probability = torch.max(point_probs).item()
            suitability_score = score_map[z, y, x].item()
            
            prompt_info = {
                'rank': i + 1,
                'coords_3d': coords_3d,
                'predicted_class': predicted_class,
                'target_class': class_id,
                'class_probability': class_probability,
                'suitability_score': suitability_score,
                'all_class_probs': point_probs.tolist()
            }
            
            class_prompts[class_id].append(prompt_info)
    
    return class_prompts

if __name__ == "__main__":
    # 运行带可视化的测试
    class_prompts, tensors_dict, fig1, fig2 = DecoupledUncertaintyGenerator_test_with_visualization()
    
    print("\n3D可视化完成！")
    print("图1: 3D体积和提示点可视化")
    print("图2: 详细的提示点分析")