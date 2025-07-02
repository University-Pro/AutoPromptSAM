"""
Amos数据集dataloader
读取的数据BCHWD
"""

import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Tuple, Dict

class AmosConfig:
    def __init__(
        self,
        save_dir: str = "./datasets/Amos",  # 修改为更合理的默认路径
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        num_classes: int = 16,
    ):
        self.save_dir = save_dir
        self.patch_size = patch_size
        self.num_classes = num_classes

class AmosDataset(Dataset):
    def __init__(
        self,
        split: str = 'train',
        config: AmosConfig = AmosConfig(),
        transform=None,
        preload: bool = False,
        training_num: int = None
    ):
        # 初始化配置
        self.config = config
        
        # 读取数据列表
        self.ids_list = self._read_list(split)
        
        # 应用样本数量限制
        if training_num is not None and training_num > 0:
            self.ids_list = self.ids_list[:training_num]
        
        # 预加载设置
        self.preload = preload
        self.data_cache = {}
        if preload:
            self._preload_data()
        
        self.transform = transform

    def _read_list(self, split: str) -> list:
        """读取数据分割列表"""
        list_path = os.path.join(self.config.save_dir, f"{split}.txt")
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Split file {list_path} not found")
        
        ids = np.loadtxt(list_path, dtype=str).tolist()
        return sorted(ids)

    def _preload_data(self):
        """预加载数据到内存"""
        for data_id in tqdm(self.ids_list, desc=f"Preloading {len(self.ids_list)} samples"):
            self.data_cache[data_id] = self._load_sample(data_id)

    def _load_sample(self, data_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载单个样本"""
        img_path = os.path.join(self.config.save_dir, 'data', f'{data_id}_image.npy')
        label_path = os.path.join(self.config.save_dir, 'data', f'{data_id}_label.npy')
        
        # 检查文件存在性
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_path} not found")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label {label_path} not found")
        
        # 加载并预处理数据
        image = np.load(img_path)
        label = np.load(label_path)
        
        # 维度转换 DHW -> HWD，保持和LA数据集一致
        image = np.transpose(image, (1, 2, 0))  # (D,H,W) -> (H,W,D)
        label = np.transpose(label, (1, 2, 0))  # (D,H,W) -> (H,W,D)
        
        # 归一化处理
        image = np.clip(image, -75, 275)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return image.astype(np.float32), label.astype(np.int64)

    def __len__(self) -> int:
        return len(self.ids_list)

    def __getitem__(self, index: int) -> Dict:
        data_id = self.ids_list[index]
        
        # 获取数据
        if self.preload:
            image, label = self.data_cache[data_id]
        else:
            image, label = self._load_sample(data_id)
        
        # 构建样本字典
        sample = {
            'image': image,
            'label': label,
            'id': data_id
        }
        
        # 数据增强
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def analyze_dataset(split: str, training_num: int = None):
    """数据集分析工具（带错误处理）"""
    try:
        config = AmosConfig(save_dir="./datasets/Amos")  # 显式指定路径
        dataset = AmosDataset(
            split=split,
            config=config,
            training_num=training_num
        )
        
        # 获取第一个样本
        sample = dataset[0]
        
        print(f"\nAMOS {split.upper()} 数据集分析结果:")
        print(f"- 总样本数: {len(dataset)}")
        print(f"- 图像尺寸: {sample['image'].shape[1:]}")  # 忽略通道维度
        print(f"- 标签尺寸: {sample['label'].shape}")
        print(f"- 像素值范围: [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
        print(f"- 唯一标签值: {np.unique(sample['label'])}")
        print("-"*60)
        
    except Exception as e:
        print(f"\n分析{split}数据集时出错: {str(e)}")

if __name__ == "__main__":
    # 测试所有分割数据集
    for split in ['train', 'test', 'eval']:
        analyze_dataset(split, training_num=0)  # training_num=0表示不限制样本数量