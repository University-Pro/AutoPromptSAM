"""
使用3D卷积的UNet3D
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# 定义3D UNet的编码器部分
class UNet3DEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3DEncoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        pooled = self.pool(x)
        return x, pooled

# 定义3D UNet的解码器部分
class UNet3DDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3DDecoder, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

# 定义完整的3D UNet网络
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        # 编码器部分
        self.encoder1 = UNet3DEncoder(in_channels, 64)
        self.encoder2 = UNet3DEncoder(64, 128)
        self.encoder3 = UNet3DEncoder(128, 256)
        self.encoder4 = UNet3DEncoder(256, 512)

        # 中间部分
        self.middle_conv1 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        self.middle_conv2 = nn.Conv3d(1024, 1024, kernel_size=3, padding=1)

        # 解码器部分
        self.decoder4 = UNet3DDecoder(1024, 512)
        self.decoder3 = UNet3DDecoder(512, 256)
        self.decoder2 = UNet3DDecoder(256, 128)
        self.decoder1 = UNet3DDecoder(128, 64)

        # 最后的1x1卷积层，用于将通道数映射到输出类别数
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器部分
        x1, pooled1 = self.encoder1(x)
        x2, pooled2 = self.encoder2(pooled1)
        x3, pooled3 = self.encoder3(pooled2)
        x4, pooled4 = self.encoder4(pooled3)

        # 中间部分
        x = F.relu(self.middle_conv1(pooled4))
        x = F.relu(self.middle_conv2(x))

        # 解码器部分
        x = self.decoder4(x, x4)
        x = self.decoder3(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x1)

        # 最后的1x1卷积层
        x = self.final_conv(x)
        return x

# 主函数
def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入 (batch_size, channels, depth, height, width)
    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device=device)

    # 初始化3D UNet模型
    model = UNet3D(in_channels=1, out_channels=2).to(device=device)

    # 使用summary查看模型的参数量和计算复杂度
    summary(model=model, input_size=(1, 1, 112, 112, 80), device=device)

    # 前向传播
    output_tensor = model(input_tensor)

    # 打印输入和输出的形状
    print("输入形状:", input_tensor.shape)
    print("输出形状:", output_tensor.shape)

if __name__ == "__main__":
    main()