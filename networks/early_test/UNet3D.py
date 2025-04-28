"""
UNet网络
不使用3D卷积
而是采用切片的方法查看UNet在3D数据集上的效果
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# 双层卷积
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# 下采样部分
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# 上采样部分
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels=out_channels * 2, out_channels=out_channels)

    # 设计跳连，其中x2是跳连传递进来的
    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 适配尺寸
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 合并上采样的特征图和跳跃连接的特征图
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 最后的输出卷积，卷积核为1
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 下面构建2D UNet的各个模块
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)  

        self.up1 = Up(1024, 512)      
        self.up2 = Up(512, 256)      
        self.up3 = Up(256, 128)       
        self.up4 = Up(128, 64)        

        self.outc = OutConv(64, n_classes)

    def slice_3d_to_2d(self, x):
        """
        将3D张量切片为2D张量列表
        输入x的形状为 (B, C, H, W, D)
        输出为长度为D的列表，每个元素形状为 (B, C, H, W)
        """
        slices = []
        for depth in range(x.size(-1)):  # 遍历深度维度
            slices.append(x[..., depth])  # 等价于 x[:, :, :, :, depth]
        return slices

    def stack_2d_to_3d(self, slices):
        """
        将2D张量列表重新堆叠为3D张量
        输入列表中每个元素的形状为 (B, C, H, W)，
        输出张量形状为 (B, C, H, W, D)
        """
        stacked = torch.stack(slices, dim=-1)  # 在最后一维堆叠切片
        return stacked

    def _forward2d(self, x):
        """
        定义针对2D输入（4D张量）的前向传播
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits

    def forward(self, x):
        """
        支持2D和3D输入：
         - 如果输入为4D张量 (B, C, H, W)，则直接使用2D网络进行处理；
         - 如果输入为5D张量 (B, C, H, W, D)，则对深度维度进行切片，
           分别对每个2D切片进行前向传播，再将所有切片的结果堆叠成3D输出。
        """
        if x.dim() == 5:
            # 3D输入，形状为 (B, C, H, W, D)
            slices = self.slice_3d_to_2d(x)  # 得到列表，每个元素形状为 (B, C, H, W)
            outputs = []
            for sl in slices:
                out_sl = self._forward2d(sl)  # 对每个2D切片进行处理
                outputs.append(out_sl)
            # 将所有2D输出堆叠回3D，输出形状为 (B, n_classes, H, W, D)
            logits = self.stack_2d_to_3d(outputs)
        else:
            # 默认认为输入为4D： (B, C, H, W)
            logits = self._forward2d(x)
        return logits

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建UNet实例，假设输入通道数为1，输出类别数为2
    model = UNet(n_channels=1, n_classes=2)

    # 测试2D输入：形状 (B, C, H, W)
    x2d = torch.randn(1, 1, 112, 112)
    output2d = model(x2d)
    print("2D输出形状:", output2d.shape)  # 应该为 (1, 2, 112, 112)

    # 测试3D输入：形状 (B, C, H, W, D)
    x3d = torch.randn(1, 1, 112, 112, 80)
    output3d = model(x3d)
    print("3D输出形状:", output3d.shape)  # 应该为 (1, 2, 112, 112, 80)

    summary(model, input_size=(1, 1, 112, 112, 80), device='cpu')