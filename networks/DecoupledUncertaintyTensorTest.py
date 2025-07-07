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
def create_sphere_tensor(
    tensor_size=(1, 1, 112, 112, 80),
    device='cuda',
    sphere_center=(56, 56, 40),  # 球心位置（在体积中心）
    sphere_radius=15,          # 球体半径
    noise_level=0.1,           # 背景噪声强度
    foreground_brightness=1.0   # 前景亮度
):
    """
    创建一个包含球形前景的三维张量，并添加背景噪声。

    Args:
        tensor_size (tuple): 张量维度 (batch_size, channels, height, width, depth)
        device (str): 设备类型（'cuda' 或 'cpu'）
        sphere_center (tuple): 球心坐标 (x, y, z)
        sphere_radius (int): 球体半径
        noise_level (float): 背景噪声强度，范围在 [0, 1]
        foreground_brightness (float): 前景亮度，范围在 [0, 1]

    Returns:
        torch.Tensor: 包含球形前景和背景噪声的张量
    """
    # 创建全零的背景张量
    dummy_input = torch.zeros(tensor_size).to(device)
    
    # 球心坐标
    center_x, center_y, center_z = sphere_center
    
    # 填充球体区域
    for x in range(center_x - sphere_radius, center_x + sphere_radius):
        for y in range(center_y - sphere_radius, center_y + sphere_radius):
            for z in range(center_z - sphere_radius, center_z + sphere_radius):
                # 计算当前点到球心的距离
                dx = x - center_x
                dy = y - center_y
                dz = z - center_z
                distance_sq = dx**2 + dy**2 + dz**2
                
                # 如果在球体内，设置为前景亮度
                if distance_sq <= sphere_radius**2:
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
            
            # 设置VNet输出，第一个为输出结果，第二个为置信度，其他的不管
            vnet_output = self.network(x)
            logits = vnet_output[0]
            log_aleatoric_var = vnet_output[1]

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
                                suppression_radius: int = 5):
    
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
        
        for i in range(k_per_class):
            best_idx = torch.argmax(temp_score_map)
            best_score = temp_score_map.reshape(-1)[best_idx]
            
            if best_score == -float('inf'):
                # print(f"类别 {class_id} 只能找到 {i} 个有效点")
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
    
    return class_prompts

def DecoupledUncertaintyGenerator_test():
    # --- 1. 准备工作 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练网络
    trained_vnet_aleatoric = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True)
    trained_vnet_aleatoric.load_state_dict(torch.load('result/VNet_Multi/LA_16/Pth/Best.pth', map_location=device))
    trained_vnet_aleatoric.to(device)

    # --- 2. 实例化新的不确定性生成器 ---
    uncertainty_gen = DecoupledUncertaintyGenerator(network=trained_vnet_aleatoric, num_mc_samples=25)

    # --- 3. 获取不确定性地图 ---
    dummy_input = create_sphere_tensor(
        tensor_size=(1, 1, 112, 112, 80),
        device=device,
        sphere_center=(56, 56, 40),  # 球心位置（在体积中心）
        sphere_radius=15,          # 球体半径
        noise_level=0.1,           # 背景噪声强度
        foreground_brightness=1.0   # 前景亮度
    )
    H_epistemic, H_aleatoric, final_segmentation_probs = uncertainty_gen(dummy_input)

    print(f"模型不确定性 H_epistemic shape: {H_epistemic.shape}")
    print(f"数据不确定性 H_aleatoric shape: {H_aleatoric.shape}")
    print(f'最终分割概率图 final_segmentation_probs shape: {final_segmentation_probs.shape}')

    # --- 4. 计算Prompt适宜度分数 S ---
    w_epistemic = 0.5
    w_aleatoric = 0.5

    def normalize(tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

    H_epistemic_norm = normalize(H_epistemic)
    H_aleatoric_norm = normalize(H_aleatoric)

    # 计算分数 S (我们想要S越大越好)
    prompt_suitability_score = w_epistemic * (1 - H_epistemic_norm) + w_aleatoric * (1 - H_aleatoric_norm)
    print(f"Prompt分数地图 shape: {prompt_suitability_score.shape}")

    # --- 5. 为每个类别使用非极大值抑制找到最佳K个Prompt点 ---
    # 转换格式
    score_map = prompt_suitability_score.squeeze(0).squeeze(0).permute(2, 0, 1)  # [D, H, W]
    segmentation_probs = final_segmentation_probs.squeeze(0).permute(0, 1, 2, 3)  # [C, H, W, D]
    
    print(f"Score map shape for NMS: {score_map.shape}")
    print(f"Segmentation probs shape: {segmentation_probs.shape}")

    # 使用非极大值抑制为每个类别找到Top-K个提示点
    k_per_class = 20
    num_classes = 2
    suppression_radius = 10
    
    class_prompts = find_top_k_prompts_per_class(
        score_map=score_map,
        segmentation_probs=segmentation_probs,
        k_per_class=k_per_class,
        num_classes=num_classes,
        suppression_radius=suppression_radius
    )
    
    # --- 6. 输出每个类别的结果 ---
    print(f"\n=== 每个类别的Top-{k_per_class}最佳Prompt点 ===")
    
    all_sam_prompts = []
    all_results = []
    
    for class_id in range(num_classes):
        prompts = class_prompts.get(class_id, [])
        print(f"\n--- 类别 {class_id} ({len(prompts)} 个点) ---")
        
        for i, prompt in enumerate(prompts[:10]):  # 只显示前10个以节省空间
            y, x, z = prompt['coords_3d']
            print(f"第{i+1}名: 坐标[{y}, {x}, {z}], "
                  f"类别{prompt['predicted_class']}, 概率{prompt['class_probability']:.4f}, "
                  f"分数{prompt['suitability_score']:.4f}")
        
        if len(prompts) > 10:
            print(f"    ... 还有 {len(prompts) - 10} 个点")
        
        # 为SAM模型生成提示点格式
        for prompt in prompts:
            y, x, z = prompt['coords_3d']
            sam_prompt = {
                'point_2d': [x, y],  # SAM使用 [x, y] 格式
                'point_3d': [x, y, z],
                'label': prompt['predicted_class'],  # 重要：现在包含标签
                'target_class': prompt['target_class'],  # 目标类别
                'confidence': prompt['class_probability'],
                'uncertainty_score': 1 - prompt['suitability_score']
            }
            all_sam_prompts.append(sam_prompt)
            
        all_results.extend(prompts)
    
    # --- 7. 统计信息 ---
    print(f"\n=== 统计信息 ===")
    total_points = sum(len(class_prompts.get(i, [])) for i in range(num_classes))
    print(f"总共生成了 {total_points} 个提示点")
    
    for class_id in range(num_classes):
        class_count = len(class_prompts.get(class_id, []))
        print(f"类别 {class_id}: {class_count} 个点")
    
    # --- 8. 计算每个类别内的空间分布质量 ---
    print(f"\n=== 每个类别的空间分布分析 ===")
    
    for class_id in range(num_classes):
        prompts = class_prompts.get(class_id, [])
        if len(prompts) < 2:
            continue
            
        distances = []
        for i in range(len(prompts)):
            for j in range(i + 1, len(prompts)):
                coord1 = torch.tensor(prompts[i]['coords_3d'], dtype=torch.float32)
                coord2 = torch.tensor(prompts[j]['coords_3d'], dtype=torch.float32)
                dist = torch.norm(coord2 - coord1).item()
                distances.append(dist)
        
        if distances:
            min_distance = min(distances)
            max_distance = max(distances)
            avg_distance = sum(distances) / len(distances)
            
            print(f"类别 {class_id} - 点间距离: 最小{min_distance:.2f}, 最大{max_distance:.2f}, 平均{avg_distance:.2f}")
            
            if min_distance < suppression_radius:
                print(f"⚠️ 警告: 存在距离小于抑制半径({suppression_radius})的点对")
            else:
                print(f"✅ 所有点都满足最小距离约束")
    
    # --- 9. SAM模型提示点格式输出 ---
    print(f"\n=== SAM模型提示点格式 (前10个示例) ===")
    for i, sam_prompt in enumerate(all_sam_prompts[:10]):
        print(f"点{i+1}: 2D坐标{sam_prompt['point_2d']}, 3D坐标{sam_prompt['point_3d']}, "
              f"标签{sam_prompt['label']}, 目标类别{sam_prompt['target_class']}, "
              f"置信度{sam_prompt['confidence']:.4f}")
    
    if len(all_sam_prompts) > 10:
        print(f"... 还有 {len(all_sam_prompts) - 10} 个提示点")
    
    # --- 10. 转化为SAM的输入格式 ---
    all_coords = []
    all_labels = []

    # 收集所有有效的提示点（按排名顺序）
    for class_id in range(num_classes):
        prompts = class_prompts.get(class_id, [])
        for prompt in prompts:
            # 注意：坐标顺序调整为 [x, y, z]（与SAM的坐标系一致）
            x, y, z = prompt['coords_3d']
            all_coords.append([x, y, z])
            # 使用预测类别作为标签（0或1）
            all_labels.append(prompt['predicted_class'])

    # 转换为张量
    coords_tensor = torch.tensor(all_coords, dtype=torch.float32)  # [27, 3]
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)     # [27]
    
    # 添加批次维度
    coords_tensor = coords_tensor.unsqueeze(0)  # [1, 400, 3]
    labels_tensor = labels_tensor.unsqueeze(0)  # [1, 400]

    print(f"\nSAM输入张量:")
    print(f"coords shape: {coords_tensor.shape}")
    print(f"labels shape: {labels_tensor.shape}")

    return {
        'class_prompts': class_prompts,
        'sam_prompts': all_sam_prompts,
        'total_points': total_points,
        'spatial_stats': {
            'suppression_radius': suppression_radius,
            'k_per_class': k_per_class,
            'num_classes': num_classes
        },
        'sam_coords': coords_tensor,
        'sam_labels': labels_tensor,
    }

if __name__ == "__main__":
    results = DecoupledUncertaintyGenerator_test()
    
    # 可以进一步处理结果
    print(f"\n=== 最终总结 ===")
    print(f"成功为 {results['spatial_stats']['num_classes']} 个类别生成了总共 {results['total_points']} 个高质量提示点")
    print(f"每个类别最多 {results['spatial_stats']['k_per_class']} 个点")
    print(f"所有点都包含了对应的类别标签信息")
    print(f"这些点基于不确定性分析，具有良好的空间分布")
    print(f"可以直接用于SAM模型的提示输入")