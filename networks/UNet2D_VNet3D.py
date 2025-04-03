import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# 指定GPU设备
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定第0块GPU

# 导入UNet、VNet网络
# from UNet import UNet
from UNet3D import UNet
from VNet import VNet

class UNet2D_VNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet2D_VNet3D, self).__init__()
        self.unet = UNet(n_channels=n_channels, n_classes=n_classes)  # 2D UNet分支
        self.vnet = VNet(n_channels=n_channels, n_classes=n_classes, n_filters=16, has_dropout=True)  # 3D VNet分支

    def forward(self, x):
        # print(f'Input x shape: {x.shape}')
        
        # 使用VNet处理3D张量
        vnet_output = self.vnet(x)  # (1, n_classes, 112, 112, 80)

        # 使用UNet处理张量
        unet_output = self.unet(x)

        return vnet_output, unet_output


if __name__ == "__main__":

    # 检查加速功能是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 初始化模型并将其移动到 GPU（如果可用）
    model = UNet2D_VNet3D(n_channels=1, n_classes=2).to(device)
    input_tensor = torch.randn(1, 1, 112, 112, 80).to(device)  # 输入3D张量

    # 前向传播，获取VNet和UNet的输出
    output_vnet, output_unet = model(input_tensor)
    print("VNet Output shape:", output_vnet.shape)
    print("UNet Output shape:", output_unet.shape)

    summary(model=model, input_size=(1, 1, 112, 112, 80))  # 打印模型结构信息

    print(model)  # 打印模型结构