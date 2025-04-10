"""
利用粗分割为没有Prompt的生成Prompt网络
使用3D数据集
粗分割模块添加预训练权重，获得粗分割结果，然后获得对应的点Prompt
这个粗分割结果可以作为无标签数据的一个伪标签之一
获取的是点Prompt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from torchinfo import summary
import numpy as np

# 导入SAM的相关模块
from networks.sam_med3d.modeling.image_encoder3D import ImageEncoderViT3D
from networks.sam_med3d.modeling.prompt_encoder3D import PromptEncoder3D   
# 导入粗分割网络
from networks.VNet import VNet

# 从粗分割输出中提取点提示的函数
def extract_point_prompts(output_tensor, num_points_per_class=5, threshold=0.5):
    """
    从粗分割的输出张量中提取点提示。
    参数:
        output_tensor: 粗分割输出，形状为 (B, C, H, W, D)，其中 C 是类别数
        num_points_per_class: 每个类别采样的点数
        threshold: 概率阈值，用于确定正点
    返回:
        points: (coords, labels) 的元组，coords 形状为 (B, N, 3)，labels 形状为 (B, N)
    """
    batch_size = output_tensor.shape[0]
    num_classes = output_tensor.shape[1]
    device = output_tensor.device
    
    # 对输出进行 softmax 归一化，得到概率图
    prob_tensor = F.softmax(output_tensor, dim=1)
    
    coords_list = []
    labels_list = []
    
    for b in range(batch_size):
        batch_coords = []
        batch_labels = []
        
        # 对每个类别进行处理（假设类别 0 是背景，1 是前景类别）
        for c in range(1, num_classes):  # 跳过背景类别 (c=0)
            # 获取该类别的概率图
            prob_map = prob_tensor[b, c]
            # 找到概率高于阈值的点作为正点
            fg_indices = torch.where(prob_map > threshold)
            if len(fg_indices[0]) > 0:
                fg_sample_idx = torch.randperm(len(fg_indices[0]))[:num_points_per_class]
                fg_coords = torch.stack([fg_indices[0][fg_sample_idx], 
                                       fg_indices[1][fg_sample_idx], 
                                       fg_indices[2][fg_sample_idx]], dim=-1)
                fg_labels = torch.ones(len(fg_sample_idx), dtype=torch.long, device=device)
                batch_coords.append(fg_coords)
                batch_labels.append(fg_labels)
            
            # 从背景类别 (c=0) 中采样负点
            if c == 1:  # 只在第一个类别时采样背景点，避免重复
                bg_prob_map = prob_tensor[b, 0]
                bg_indices = torch.where(bg_prob_map > threshold)
                if len(bg_indices[0]) > 0:
                    bg_sample_idx = torch.randperm(len(bg_indices[0]))[:num_points_per_class]
                    bg_coords = torch.stack([bg_indices[0][bg_sample_idx], 
                                           bg_indices[1][bg_sample_idx], 
                                           bg_indices[2][bg_sample_idx]], dim=-1)
                    bg_labels = torch.zeros(len(bg_sample_idx), dtype=torch.long, device=device)
                    batch_coords.append(bg_coords)
                    batch_labels.append(bg_labels)
        
        if batch_coords:
            batch_coords = torch.cat(batch_coords, dim=0)
            batch_labels = torch.cat(batch_labels, dim=0)
        else:
            # 如果没有点，创建空张量
            batch_coords = torch.zeros(0, 3, dtype=torch.float32, device=device)
            batch_labels = torch.zeros(0, dtype=torch.long, device=device)
        
        coords_list.append(batch_coords)
        labels_list.append(batch_labels)
    
    # 堆叠为批量张量
    max_points = max([c.shape[0] for c in coords_list]) if coords_list else 0
    if max_points > 0:
        coords = torch.zeros(batch_size, max_points, 3, dtype=torch.float32, device=device)
        labels = torch.zeros(batch_size, max_points, dtype=torch.long, device=device)
        for b in range(batch_size):
            num_points = coords_list[b].shape[0]
            if num_points > 0:
                coords[b, :num_points] = coords_list[b]
                labels[b, :num_points] = labels_list[b]
            # 剩余的点用 -1 填充（表示非点）
            if num_points < max_points:
                labels[b, num_points:] = -1
    else:
        coords = torch.zeros(batch_size, 1, 3, dtype=torch.float32, device=device)
        labels = torch.full((batch_size, 1), -1, dtype=torch.long, device=device)
    
    return (coords, labels)

# 测试函数
def RoughSegmentation_Prompt_Test(pretrained_weights_path=None):
    """
    测试粗分割网络并生成点提示。
    参数:
        pretrained_weights_path: 预训练权重的路径，若为 None 则不加载
    返回:
        sparse_embeddings, dense_embeddings: PromptEncoder3D 的输出
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 设置模拟输入张量（这里是 3D 数据，分辨率为 112x112x80）
    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device=device)
    
    # 初始化粗分割网络为 VNet
    roughsegnetwork = VNet(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True).to(device)
    
    # 加载预训练权重（如果提供了路径）
    if pretrained_weights_path is not None:
        try:
            state_dict = torch.load(pretrained_weights_path, map_location=device)
            roughsegnetwork.load_state_dict(state_dict)
            print(f"Loaded pretrained weights from {pretrained_weights_path}")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
    
    # 设置为评估模式
    roughsegnetwork.eval()
    
    # 通过粗分割网络
    with torch.no_grad():
        output_tensor = roughsegnetwork(input_tensor)
    print(f"VNet Output shape: {output_tensor.shape}")  # 输出结果为 [1, 2, 112, 112, 80]
    
    # 从输出张量中提取点提示
    points = extract_point_prompts(output_tensor, num_points_per_class=5, threshold=0.5)
    coords, labels = points
    print(f"Point coords shape: {coords.shape}, Labels shape: {labels.shape}")
    
    # 初始化 PromptEncoder3D，调整输入尺寸以匹配实际数据
    prompt_encoder = PromptEncoder3D(
        embed_dim=255,                      # 嵌入维度，这里不能写256而是要255
        image_embedding_size=(14, 14, 10),  # 图像嵌入的空间尺寸，假设下采样率为 8
        input_image_size=(112, 112, 80),    # 输入图像尺寸，与实际输入匹配
        mask_in_chans=16,                   # 掩码编码的隐藏通道数
        activation=nn.GELU
    ).to(device)
    
    # 将点提示输入到 PromptEncoder3D
    try:
        sparse_embeddings, dense_embeddings = prompt_encoder(points=points, boxes=None, masks=None)
        print(f"Sparse embeddings shape (points): {sparse_embeddings.shape}")
        print(f"Dense embeddings shape: {dense_embeddings.shape}")
    except Exception as e:
        print(f"Error in prompt encoding: {e}")
        sparse_embeddings = None
        dense_embeddings = None
    
    return sparse_embeddings, dense_embeddings

if __name__ == "__main__":
    # 测试粗分割网络
    pretrained_weights_path = None  # 替换为实际路径，例如 "path/to/pretrained/weights.pth"
    sparse_emb, dense_emb = RoughSegmentation_Prompt_Test(pretrained_weights_path)