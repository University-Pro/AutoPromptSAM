"""
测试MaskDecoder模块
测试PromptEncoder模块
"""
import torch
from torch import nn as nn
from networks.sam_med3d.modeling.mask_decoder3D import MaskDecoder3D
from networks.sam_med3d.modeling.prompt_encoder3D import PromptEncoder3D

def MaskDecodker3D_Test():
    """
    测试MaskDecoder3D模块
    """
    model = MaskDecoder3D(
        transformer_dim=256,
        num_multimask_outputs=3,
        activation=nn.GELU,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )

    # 模拟输入
    B, C, X, Y, Z = 2, 256, 16, 16, 16
    N = 5  # 稀疏提示数量
    image_embeddings = torch.randn(B, C, X, Y, Z)
    print(f'image_embeddings.shape = {image_embeddings.shape}')  # 应为 (B, C, X, Y, Z)
    image_pe = torch.randn(B, C, X, Y, Z)
    print(f'image_pe.shape = {image_pe.shape}')  # 应为 (B, C, X, Y, Z)
    sparse_prompt_embeddings = torch.randn(B, N, C)
    print(f'sparse_prompt_embeddings.shape = {sparse_prompt_embeddings.shape}')  # 应为 (B, N, C)
    dense_prompt_embeddings = torch.randn(B, C, X, Y, Z)
    print(f'dense_prompt_embeddings.shape = {dense_prompt_embeddings.shape}')  # 应为 (B, C, X, Y, Z)
    multimask_output = True

    # 前向传播
    masks, iou_pred = model(
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
    )

    # 检查输出形状
    print(f"masks.shape = {masks.shape}")  # 应为 (B, num_multimask_outputs, X, Y, Z)
    print(f"iou_pred.shape = {iou_pred.shape}")  # 应为 (B, num_multimask_outputs)

def PromptEncoder_Test():
    # 初始化模型
    model = PromptEncoder3D(
        embed_dim=256,
        image_embedding_size=(14, 14, 10),
        input_image_size=(112, 112, 80),
        mask_in_chans=16,
        activation=nn.GELU,
    )

    # 1. 模拟点提示 (B=1, N=2个点)
    points_coords = torch.tensor([[[10, 20, 30], [40, 50, 60]]])  # (1, 2, 3)
    points_labels = torch.tensor([[1, 0]])  # 第一个点是正点，第二个是负点

    # 2. 模拟框提示 (B=1, 1个框)
    boxes = torch.tensor([[[10, 20, 30, 70, 80, 90]]])  # (x1,y1,z1,x2,y2,z2)

    # 3. 模拟掩码输入 (B=1, 1, 112, 112, 80)
    masks = torch.randn(1, 1, 112, 112, 80)  # 随机初始化

    # 模拟输入
    points = (points_coords, points_labels)  # 点坐标 + 标签
    boxes = boxes  # 框坐标
    masks = masks  # 掩码

    # 前向传播
    sparse_embeddings, dense_embeddings = model(points, boxes, masks)

    # 检查输出形状
    print(f"sparse_embeddings.shape = {sparse_embeddings.shape}")  # 应为 (1, 4, 256)
    print(f"dense_embeddings.shape = {dense_embeddings.shape}")    # 应为 (1, 256, 14, 14, 10)

if __name__=="__main__":
    # print("Testing MaskDecoder3D...")
    # MaskDecodker3D_Test()
    # print("\nTesting PromptEncoder3D...")
    PromptEncoder_Test()