import os
import sys
import imageio as iio
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

sys.path.append('/samba/network-storage/ssd/home/pi/sam2-test')  # 设置运行目录

# 定义训练集和测试集的默认变换操作
def get_transforms(image_size):
    """
    返回训练和测试的变换操作
    :param image_size: 图像的目标大小 (height, width)
    """
    train_transforms = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(image_size, padding=10),  # padding用于补齐旋转后的空白区域
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
    ])
    return train_transforms, test_transforms


# 自定义数据集类
class myDataSet(Dataset):
    def __init__(self, ids, path_images, path_masks, transforms=None):
        """
        初始化数据集
        :param ids: 图片 ID 列表
        :param path_images: 图片所在的文件夹路径
        :param path_masks: 掩码所在的文件夹路径
        :param transforms: 用于数据增强的变换操作
        """
        self.ids = ids
        self.path_images = path_images
        self.path_masks = path_masks
        self.transforms = transforms

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.ids)

    def __getitem__(self, index):
        """
        获取指定索引的图像和掩码数据
        """
        # 获取当前图像和掩码的路径
        path_img = os.path.join(self.path_images, self.ids[index])
        path_mask = os.path.join(self.path_masks, self.ids[index])

        # 读取图像和掩码，并归一化到 [0, 1] 的范围
        img = iio.v3.imread(path_img) / 255
        mask = iio.v3.imread(path_mask)[:, :, 0] / 255
        mask = mask.round()  # 将掩码值二值化为 0 或 1

        # 转换为 PyTorch 张量并调整维度
        img = torch.FloatTensor(np.transpose(img, [2, 0, 1]))  # [C, H, W]
        mask = torch.FloatTensor(mask).unsqueeze(0)  # [1, H, W]

        # 应用数据增强（如果存在）
        if self.transforms:
            sample = torch.cat((img, mask), 0)  # 将图像和掩码拼接为一个张量
            sample = self.transforms(sample)  # 应用变换操作
            img = sample[:img.shape[0], ...]
            mask = sample[img.shape[0]:, ...]

        return img, mask


# 主函数，包含数据集测试逻辑
if __name__ == "__main__":
    # 设置动态参数
    IMAGE_SIZE = (512, 512)  # 动态控制图像大小
    TRAIN_LIST_PATH = "./datasets/Kvasir_Source/list/train.txt"
    VAL_LIST_PATH = "./datasets/Kvasir_Source/list/val.txt"
    IMAGE_PATH = "./datasets/Kvasir_Source/images"
    MASK_PATH = "./datasets/Kvasir_Source/masks"

    # 加载数据增强变换
    train_transforms, test_transforms = get_transforms(IMAGE_SIZE)

    # 加载训练集和验证集的文件 ID
    with open(TRAIN_LIST_PATH, 'r') as f:
        ids_train = [l.strip() + '.jpg' for l in f]
    with open(VAL_LIST_PATH, 'r') as f:
        ids_val = [l.strip() + '.jpg' for l in f]

    # 创建训练集和验证集对象，传入对应的变换操作
    custom_dataset_train = myDataSet(ids_train, IMAGE_PATH, MASK_PATH, transforms=train_transforms)
    custom_dataset_val = myDataSet(ids_val, IMAGE_PATH, MASK_PATH, transforms=test_transforms)

    # 打印数据集的大小
    print(f"Train Dataset size: {len(custom_dataset_train)}")
    print(f"Validation Dataset size: {len(custom_dataset_val)}")

    # 读取并打印训练集的一些样本信息
    print("---- Train Dataset Samples ----")
    for i in range(3):  # 读取前3个样本
        img, mask = custom_dataset_train[i]
        print(f"Sample {i+1} - Image shape: {img.shape}, Mask shape: {mask.shape}")

    # 读取并打印验证集的一些样本信息
    print("---- Validation Dataset Samples ----")
    for i in range(3):  # 读取前3个样本
        img, mask = custom_dataset_val[i]
        print(f"Sample {i+1} - Image shape: {img.shape}, Mask shape: {mask.shape}")
