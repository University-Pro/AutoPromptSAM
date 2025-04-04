"""
LA数据集的半监督读取办法
不使用双流编码器
而是分别读取有标签的数据和无标签的数据
"""
from torch.utils.data import Dataset
import h5py
import os
import numpy as np
import nibabel as nib  # 添加nibabel库用于处理nii格式
from torchvision import transforms  # 图像转换

# 导入数据增强
from utils.ImageAugment import RandomRotFlip_LA as RandomRotFlip
from utils.ImageAugment import RandomCrop_LA as RandomCrop
from utils.ImageAugment import ToTensor_LA as ToTensor

class LAHeart(Dataset):
    """ LA Dataset (Semi-Supervised Version) """
    def __init__(self, base_dir=None, split='train_label', num=None, transform=None):
        self._base_dir = base_dir  # 数据集根路径
        self.transform = transform  # 数据预处理
        self.image_list = []  # 样本列表

        # 验证split参数有效性
        valid_splits = ['train_label', 'train_unlabel', 'test']
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")

        # 构建对应的list文件路径
        list_path = os.path.join(self._base_dir, f"{split}.list")
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"List file {list_path} not found.")

        # 读取样本列表
        with open(list_path, 'r') as f:
            self.image_list = f.readlines()

        # 清理换行符并过滤空行
        self.image_list = [item.strip() for item in self.image_list if item.strip()]
        
        # 裁剪到指定数量样本
        if num is not None:
            self.image_list = self.image_list[:num]

        print(f"Total {len(self.image_list)} samples in '{split}' split.")

    def __len__(self):
        return len(self.image_list)

    def _get_volume_dimensions(self, idx):
        """获取指定样本的维度信息"""
        image_name = self.image_list[idx]
        h5_path = os.path.join(self._base_dir, "data", image_name, "mri_norm2.h5")
        with h5py.File(h5_path, 'r') as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:]
        return {
            'image_shape': image.shape,
            'label_shape': label.shape
        }

    def __getitem__(self, idx):
        # 根据索引读取样本
        image_name = self.image_list[idx]
        h5_path = os.path.join(self._base_dir, "data", image_name, "mri_norm2.h5")
        
        with h5py.File(h5_path, 'r') as h5f:
            image = h5f['image'][:]  # 读取图像
            label = h5f['label'][:]  # 读取标签

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample
    
# 在测试函数中添加验证代码
def LA_dataset_test():
    patch_size = (112, 112, 80)
    
    # 加载训练数据集
    db_train = LAHeart(base_dir="./datasets/LA",
                        split='train_unlabel',
                        transform=transforms.Compose([  # 定义数据预处理
                            RandomRotFlip(),
                            RandomCrop(patch_size),
                            ToTensor(),
                        ]),
                        num=80)
    
    # 打印数据集大小
    print(f"Dataset size: {len(db_train)}")
    
    # 打印前几个样本的形状
    for idx in range(min(5, len(db_train))):  # 测试前5个样本
        sample = db_train[idx]
        print(f"Sample {idx}:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Label shape: {sample['label'].shape}")

if __name__ == "__main__":
    LA_dataset_test()