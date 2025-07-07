import torch
import torch.nn as nn
import torch.nn.functional as F

class CARF(nn.Module):
    """
    跨编码器注意力精炼与融合模块 (Cross-Encoder Attention Rectification and Fusion, CARF)
    
    该模块用于融合来自两个不同编码器的最后一个层级的3D特征图。
    它能够处理不同的输入张量格式，并使用基于相似度的注意力机制来智能地融合特征。
    """
    def __init__(self, dim_a, dim_b, common_dim=256):
        """
        Args:
            dim_a (int): 编码器A的特征通道数。
            dim_b (int): 编码器B的特征通道数。
            common_dim (int): 内部计算时使用的公共通道维度。
        """
        super().__init__()
        
        # 投影层，将两个输入特征投影到相同的维度
        self.proj_a = nn.Conv3d(dim_a, common_dim, kernel_size=1, bias=False)
        self.proj_b = nn.Conv3d(dim_b, common_dim, kernel_size=1, bias=False)
        
        # 最终融合后的处理层，将通道数恢复或调整到期望值
        # 这里我们将融合后的特征（通道为common_dim）再通过一个卷积层
        # 这增加了模型的学习能力，也可以用来调整最终输出的通道数
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(common_dim, common_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(common_dim),
            nn.ReLU(inplace=True)
        )
        
        print("CARF Module Initialized.")
        print(f"  Encoder A input channels: {dim_a}")
        print(f"  Encoder B input channels: {dim_b}")
        print(f"  Internal common channels: {common_dim}")

    def forward(self, feature_a, feature_b):

        # --- 1. 输入格式统一 ---
        # 验证输入维度
        assert feature_a.dim() == 5, "Feature A must be 5D"
        assert feature_b.dim() == 5, "Feature B must be 5D"
        
        # 将 feature_a 从 BCHWD -> BCDHW (PyTorch standard)
        # torch.Size([1, 64, 56, 56, 40]) -> torch.Size([1, 64, 40, 56, 56])
        f_a_std = feature_a.permute(0, 1, 4, 2, 3) 
        
        # 将 feature_b 从 BHWDC -> BCDHW (PyTorch standard)
        # torch.Size([1, 56, 56, 40, 96]) -> torch.Size([1, 96, 40, 56, 56])
        f_b_std = feature_b.permute(0, 4, 3, 1, 2)
        
        # --- 2. 共享维度投影 ---
        # f_a_proj, f_b_proj 的形状都将是 (B, common_dim, D, H, W)
        f_a_proj = self.proj_a(f_a_std)
        f_b_proj = self.proj_b(f_b_std)
        
        # --- 3. 注意力图生成 ---
        # sim_map 的形状是 (B, D, H, W)
        sim_map = F.cosine_similarity(f_a_proj, f_b_proj, dim=1)
        
        # 使用 sigmoid 将相似度 (-1, 1) 映射到 (0, 1) 作为门控权重
        # attention_map 形状 (B, 1, D, H, W) 以便进行广播乘法
        attention_map = torch.sigmoid(sim_map).unsqueeze(1)
        
        # --- 4. 双向特征精炼 ---
        # f_a_refined: 编码器A特征中与B相似的部分
        f_a_refined = f_a_proj * attention_map
        
        # f_b_refined: 编码器B特征中与A互补(不相似)的部分
        f_b_refined = f_b_proj * (1 - attention_map)
        
        # --- 5. 最终融合 ---
        fused_feature = f_a_refined + f_b_refined
        
        # 通过最后的卷积层进一步处理融合特征
        fused_feature = self.fusion_conv(fused_feature)
        
        return fused_feature, attention_map.squeeze(1)


# --- 模拟输入与测试 ---

# 1. 定义两个编码器最后一层特征的维度
# 假设这是您提供的最后一层特征
# Feature 5 & after_encoder 
dim_A_last = 512
dim_B_last = 768
batch_size = 1

# 2. 创建模拟输入数据，严格按照您提供的格式
# 列表A的最后一个特征: BCHWD
feature_A_last = torch.randn(batch_size, dim_A_last, 7, 7, 5)

# 列表B的最后一个特征: BHWDC
feature_B_last = torch.randn(batch_size, 7, 7, 5, dim_B_last)

print("--- 模拟输入 Feature Shape ---")
print(f"Feature A (BCHWD) shape: {feature_A_last.shape}")
print(f"Feature B (BHWDC) shape: {feature_B_last.shape}")
print("-" * 30)


# 3. 实例化CARF模块
# common_dim 可以根据您的模型大小和GPU显存进行调整
carf_module = CARF(dim_a=dim_A_last, dim_b=dim_B_last, common_dim=512)
print("-" * 30)

# 4. 执行计算并获取输出
carf_module.eval()
with torch.no_grad():
    fused_output, attention = carf_module(feature_A_last, feature_B_last)

# 5. 打印结果
print("\n--- CARF 模块模拟输出 ---")
# 融合后的特征图，其 D, H, W 尺寸应保持不变
print(f"融合后特征 (Fused Feature) 的形状 (BCDH): {fused_output.shape}")
# 注意力图，其 D, H, W 尺寸也应保持不变
print(f"注意力图 (Attention Map) 的形状 (BDH): {attention.shape}")

# 我们可以检查注意力图的值
print(f"\n注意力图的值（示例）:")
print(f"  最大值: {attention.max().item():.4f}")
print(f"  最小值: {attention.min().item():.4f}")
print(f"  平均值: {attention.mean().item():.4f}")