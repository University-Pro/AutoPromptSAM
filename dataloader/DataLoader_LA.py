"""
LA数据集的Dataloader
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

# 定义LAHeart数据集
class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir  # 数据集根路径
        self.transform = transform  # 数据预处理
        self.sample_list = []  # 样本列表

        # 读取训练或测试列表文件
        train_path = self._base_dir + '/train.list'
        test_path = self._base_dir + '/test.list'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        # 清理换行符
        self.image_list = [item.strip() for item in self.image_list]
        
        # 如果指定样本数量，裁剪数据列表
        if num is not None:
            self.image_list = self.image_list[:num]

        # print(f"Total {len(self.image_list)} samples in the '{split}' split.")  # 打印样本数量

    def __len__(self):
        return len(self.image_list)

    def _get_volume_dimensions(self, idx):
        """获取指定样本的维度信息"""
        image_name = self.image_list[idx]
        with h5py.File(self._base_dir + f"/data/{image_name}/mri_norm2.h5", 'r') as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:]
        return {
            'image_shape': image.shape,
            'label_shape': label.shape
        }

    def __getitem__(self, idx):
        # 根据索引读取样本
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + f"/data/{image_name}/mri_norm2.h5", 'r')
        image = h5f['image'][:]  # 读取图像
        label = h5f['label'][:]  # 读取标签
        h5f.close()

        sample = {'image': image, 'label': label}  # 构造样本字典
        if self.transform:
            sample = self.transform(sample)  # 应用预处理

        return sample

def analyze_dataset_dimensions(dataset):
    """
    分析数据集的最小维度
    返回:
        dict: 包含最小H/W/D和所有样本的维度统计信息
    """
    min_h = min_w = min_d = float('inf')
    all_dimensions = []

    for idx in range(len(dataset)):
        dims = dataset._get_volume_dimensions(idx)
        h, w, d = dims['image_shape']
        
        # 更新最小值
        min_h = min(min_h, h)
        min_w = min(min_w, w)
        min_d = min(min_d, d)
        
        all_dimensions.append({
            'name': dataset.image_list[idx],
            'image_shape': dims['image_shape'],
            'label_shape': dims['label_shape']
        })

    return {
        'min_dimensions': (min_h, min_w, min_d),
        'total_samples': len(dataset),
        'all_dimensions': all_dimensions
    }

# 在测试函数中添加验证代码
def LA_dataset_test():
    patch_size = (112, 112, 80)
    # patch_size = (128,128,128)
    
    # 加载训练数据集
    db_train = LAHeart(base_dir="./datasets/LA",
                        split='train',
                        transform=transforms.Compose([  # 定义数据预处理
                            RandomRotFlip(),
                            RandomCrop(patch_size),
                            ToTensor(),
                        ]),
                        num=16)
    
    # 打印数据集大小
    print(f"Dataset size: {len(db_train)}")
    
    # 打印前几个样本的形状
    for idx in range(min(5, len(db_train))):  # 测试前5个样本
        sample = db_train[idx]
        print(f"Sample {idx}:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Label shape: {sample['label'].shape}")

# 添加H5到NII.GZ的转换函数
def convert_h5_to_nii(base_dir, output_dir=None, split='train', num=None):
    """
    将LA数据集中的h5格式数据转换为nii.gz格式
    
    参数:
        base_dir: 数据集根目录
        output_dir: 输出目录，默认为base_dir/nii_data
        split: 'train'或'test'
        num: 要转换的样本数量，None表示全部
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(base_dir, 'nii_data')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取样本列表
    list_file = os.path.join(base_dir, f'{split}.list')
    with open(list_file, 'r') as f:
        image_list = [item.strip() for item in f.readlines()]
    
    # 如果指定样本数量，裁剪数据列表
    if num is not None:
        image_list = image_list[:num]
    
    print(f"开始转换{len(image_list)}个{split}集样本...")
    
    # 遍历每个样本进行转换
    for i, image_name in enumerate(image_list):
        # 创建样本输出目录
        sample_output_dir = os.path.join(output_dir, image_name)
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # 读取h5文件
        h5_path = os.path.join(base_dir, 'data', image_name, 'mri_norm2.h5')
        h5f = h5py.File(h5_path, 'r')
        image = h5f['image'][:]  # 读取图像
        label = h5f['label'][:]  # 读取标签
        h5f.close()
        
        # 转换为nii.gz格式并保存
        # 创建NIfTI对象 - 使用标准仿射矩阵
        affine = np.eye(4)  # 单位矩阵作为仿射变换
        
        # 保存图像
        image_nii = nib.Nifti1Image(image, affine)
        nib.save(image_nii, os.path.join(sample_output_dir, 'image.nii.gz'))
        
        # 保存标签
        label_nii = nib.Nifti1Image(label, affine)
        nib.save(label_nii, os.path.join(sample_output_dir, 'label.nii.gz'))
        
        if (i + 1) % 10 == 0 or (i + 1) == len(image_list):
            print(f"已完成 {i+1}/{len(image_list)} 个样本的转换")
    
    print(f"转换完成！数据已保存到 {output_dir}")
    return output_dir

# 测试转换函数
def test_h5_to_nii_conversion():
    base_dir = "./datasets/LA"
    output_dir = convert_h5_to_nii(base_dir, split='train', num=5)  # 只转换5个训练样本作为测试
    print(f"转换后的数据保存在: {output_dir}")
    
    # 验证转换后的文件
    train_list_path = os.path.join(base_dir, 'train.list')
    with open(train_list_path, 'r') as f:
        image_list = [item.strip() for item in f.readlines()][:5]
    
    for image_name in image_list:
        image_path = os.path.join(output_dir, image_name, 'image.nii.gz')
        label_path = os.path.join(output_dir, image_name, 'label.nii.gz')
        
        if os.path.exists(image_path) and os.path.exists(label_path):
            # 加载nii.gz文件并打印形状
            image_nii = nib.load(image_path)
            label_nii = nib.load(label_path)
            print(f"样本 {image_name}:")
            print(f"  图像形状: {image_nii.shape}")
            print(f"  标签形状: {label_nii.shape}")
        else:
            print(f"样本 {image_name} 转换失败!")

if __name__=="__main__":
    LA_dataset_test()