import torch
import torch.nn as nn
from torchinfo import summ

class PatchMerging3D(nn.Module):
    r""" 3D Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): 输入特征的分辨率 (D, H, W)。
        dim (int): 输入通道数。
        norm_layer (nn.Module, optional): 归一化层，默认为 nn.LayerNorm。
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # 将8个相邻的3D Patch合并，通道数变为原来的8倍，然后通过线性层降维
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        """
        x: 输入张量，形状为 (B, D*H*W, C)
        """
        D, H, W = self.input_resolution
        B, L, C = x.shape

        # 常见的一些检测
        assert L == D * H * W, "输入特征的尺寸与输入分辨率不匹配"
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, f"输入分辨率 ({D}*{H}*{W}) 必须是偶数"

        # 将输入张量重塑为 (B, D, H, W, C)
        x = x.view(B, D, H, W, C)

        # 在D、H、W三个维度上每隔一个点采样，得到8个子张量
        x0 = x[:, 0::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x3 = x[:, 1::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x4 = x[:, 0::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x5 = x[:, 1::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x6 = x[:, 0::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C

        # 在通道维度上拼接，得到 (B, D/2, H/2, W/2, 8*C)
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)

        # 重塑为 (B, D/2*H/2*W/2, 8*C)
        x = x.view(B, -1, 8 * C)

        # 归一化
        x = self.norm(x)

        # 通过线性层降维，输出形状为 (B, D/2*H/2*W/2, 2*C)
        x = self.reduction(x)

        return x


def test_patch_merging_3d():
    # 输入分辨率 (D, H, W)
    input_resolution = (20, 28, 28)
    # 输入通道数
    dim = 64
    # 输入张量 (B, D*H*W, C)
    B = 1

    x = torch.randn(B, input_resolution[0] * input_resolution[1] * input_resolution[2], dim)
    print(f'input x shape is {x.shape}')

    # 创建3D Patch Merging层
    patch_merging_3d = PatchMerging3D(input_resolution, dim)
    
    # 检查占用

    # 前向传播
    output = patch_merging_3d(x)

    # 打印输入和输出的形状
    print("输出形状:", output.shape)

if __name__ == "__main__":
    # 运行测试
    test_patch_merging_3d()