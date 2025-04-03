"""
这是最基础的Unet网络
实现LA数据集进行分割
主要是用切片的方法，验证一下会损失多少
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
        """将3D张量切片为2D张量列表"""
        slices = []
        for depth in range(x.size(-1)):  # 遍历深度维度
            slices.append(x[:, :, :, :, depth])  # 提取每个切片
        return slices

    def stack_2d_to_3d(self, slices):
        """将2D张量列表重新堆叠为3D张量"""
        stacked = torch.stack(slices, dim=-1)  # 在最后一维堆叠切片
        return stacked

    def forward(self, x):
        slices = self.slice_3d_to_2d(x)  # 调用类方法

        processed_slices = []
        for slice_2d in slices:
            x1 = self.inc(slice_2d)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)


            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            
            logits = self.outc(x)
            processed_slices.append(logits)

        # 将处理后的2D切片重新堆叠为3D张量
        unet_output = self.stack_2d_to_3d(processed_slices)  # 调用类方法

        return unet_output

if __name__=="__main__":

    # 检查 加速功能 是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 初始化模型并将其移动到 GPU（如果可用）
    model = UNet(n_channels=1, n_classes=2).to(device)
    # input_tensor = torch.randn(1, 1, 112, 112).to(device)
    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device)  # 输入3D张量
    # summary(model=model,input_size=(1,1,224,224))
    print("Input shape:", input_tensor.shape)
    output = model(input_tensor)
    print("Output shape:", output.shape)
