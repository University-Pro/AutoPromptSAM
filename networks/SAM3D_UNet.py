"""
测试ImageEncoder3D模型
"""
import torch
from torchinfo import summary  # 用于打印模型结构
import torch.nn as nn
from networks.sam_med3d.modeling.image_encoder3D import ViT3DWithIntermediateOutputs as ImageEncoderViT3D
from networks.sam_med3d.modeling.mask_decoder3D import MaskDecoder3D
from networks.sam_med3d.modeling.prompt_encoder3D import PromptEncoder3D
from networks.sam_med3d.modeling.sam3D import Sam3D

def test_ImageEncoder3D():
    # 初始化3D图像编码器
    encoder = ImageEncoderViT3D(
        img_size=256,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        out_chans=256
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成模拟输入 (B=1, C=1, D=256, H=256, W=256)
    input_tensor = torch.randn(1, 1, 256, 256, 256).to(next(encoder.parameters()).device)
    
    # 打印输入尺寸
    print(f"模拟输入尺寸: {input_tensor.shape} (B,C,D,H,W)")
    
    # 前向传播
    with torch.no_grad():
        output1 = encoder(input_tensor)
    
    print(f'output1 shape is {output1.shape}')
    # print(f'output2 shape is {output2.shape}')

    # # 验证输出
    # print(f"\n编码器输出尺寸: {output.shape} (B,out_chans,D',H',W')")
    # print(f"输出值范围: [{output.min():.4f}, {output.max():.4f}]")
    
    # # 打印模型结构摘要
    # print("\n模型结构摘要:")
    # summary(
    #     encoder,
    #     input_size=(1, 1, 256, 256, 256),
    #     depth=3,
    #     col_names=["input_size", "output_size", "num_params"],
    #     device=next(encoder.parameters()).device
    # )


def test_MaskDecoder3D():
    # 初始化3D掩码解码器
    decoder = MaskDecoder3D(
        transformer_dim=384,
        num_multimask_outputs=3,
        activation=nn.GELU
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成模拟输入（尺寸需与ImageEncoderViT3D输出对齐）
    B = 1  # batch_size
    D, H, W = 16, 16, 16  # 空间维度（假设image_encoder下采样16倍）
    
    # 1. 图像嵌入 (来自ImageEncoderViT3D的输出)
    image_embeddings = torch.randn(B, 384, D, H, W).to(next(decoder.parameters()).device)
    
    # 2. 图像位置编码 (与image_embeddings同尺寸)
    image_pe = torch.randn(B, 384, D, H, W).to(next(decoder.parameters()).device)
    
    # 3. 稀疏提示嵌入 (点/框提示的嵌入)
    sparse_prompt_embeddings = torch.randn(B, 2, 384).to(next(decoder.parameters()).device)  # 假设2个点提示
    
    # 4. 密集提示嵌入 (掩码提示的嵌入)
    dense_prompt_embeddings = torch.randn(B, 384, D, H, W).to(next(decoder.parameters()).device)
    
    # 打印输入尺寸
    print("===== 模拟输入尺寸 =====")
    print(f"image_embeddings: {image_embeddings.shape} (B,embed_dim,D,H,W)")
    print(f"image_pe: {image_pe.shape}")
    print(f"sparse_prompt_embeddings: {sparse_prompt_embeddings.shape} (B,num_points,embed_dim)")
    print(f"dense_prompt_embeddings: {dense_prompt_embeddings.shape}\n")
    
    # 前向传播
    with torch.no_grad():
        masks, iou_pred = decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=True
        )
    
    # 验证输出
    print("===== 解码器输出 =====")
    print(f"masks: {masks.shape} (B,num_masks,D',H',W')")
    print(f"iou_pred: {iou_pred.shape} (B,num_masks)")
    print(f"mask值范围: [{masks.min():.4f}, {masks.max():.4f}]")
    print(f"IoU预测值示例: {iou_pred[0].tolist()}")
    
    # 打印模型结构
    print("\n===== 模型结构摘要 =====")
    summary(
        decoder,
        input_data={
            "image_embeddings": image_embeddings,
            "image_pe": image_pe,
            "sparse_prompt_embeddings": sparse_prompt_embeddings,
            "dense_prompt_embeddings": dense_prompt_embeddings,
            "multimask_output": True
        },
        depth=3,
        col_names=["input_size", "output_size", "num_params"],
        device=next(decoder.parameters()).device
    )

if __name__ == "__main__":
    test_ImageEncoder3D()
    # test_MaskDecoder3D()