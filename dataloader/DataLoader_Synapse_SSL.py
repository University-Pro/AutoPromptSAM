"""
用于Synapse半监督学习的DataLoader
"""
"""
Synapse的dataloader（支持有标签/无标签数据比例控制）
"""
import os
import sys
import random
import h5py
import numpy as np
import torch
import argparse
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from collections import defaultdict

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        
        # 保留原始sample中的所有字段
        sample = {
            'image': image,
            'label': label.long(),
            'is_labeled': sample.get('is_labeled', True),  # 保留is_labeled字段，默认为True
            'case_name': sample['case_name']
        }
        return sample

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, labeled_ratio=1.0):
        """
        Args:
            base_dir: 数据根目录
            list_dir: 包含train.txt/test.txt的目录
            split: 'train'或'test'
            transform: 数据增强
            labeled_ratio: 有标签数据的比例 (0.0-1.0)
        """
        self.transform = transform  
        self.split = split
        self.labeled_ratio = labeled_ratio
        
        # 读取所有样本
        self.all_samples = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        
        # 根据labeled_ratio划分有标签和无标签数据
        random.shuffle(self.all_samples)
        split_idx = int(len(self.all_samples) * self.labeled_ratio)
        self.labeled_samples = self.all_samples[:split_idx]
        self.unlabeled_samples = self.all_samples[split_idx:]
        
        # 合并列表，前部分是有标签，后部分是无标签
        self.sample_list = self.labeled_samples + self.unlabeled_samples
        self.is_labeled = [True]*len(self.labeled_samples) + [False]*len(self.unlabeled_samples)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        is_labeled = self.is_labeled[idx]
        slice_name = self.sample_list[idx].strip('\n')
        
        if self.split == "train":
            data_path = os.path.join(self.data_dir, "train", slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            
            # 如果是无标签数据，将label置为零矩阵
            if not is_labeled:
                label = np.zeros_like(label)
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = os.path.join(self.data_dir, "test", "{}.npy.h5".format(vol_name))
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {
            'image': image,
            'label': label,
            'is_labeled': is_labeled,
            'case_name': slice_name
        }
        
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__=="__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeled_ratio', type=float, default=1.0, 
                       help='有标签数据的比例 (0.0-1.0)')
    args = parser.parse_args()
    
    transform = RandomGenerator((512, 512))
    
    # 使用传入的labeled_ratio参数
    db_train = Synapse_dataset(
        base_dir="./datasets/Synapse/data", 
        list_dir="./datasets/Synapse/list", 
        split="train",
        transform=transform,
        labeled_ratio=args.labeled_ratio
    )
    
    print(f"总样本数: {len(db_train)}")
    print(f"有标签样本数: {sum(db_train.is_labeled)}")
    print(f"无标签样本数: {len(db_train) - sum(db_train.is_labeled)}")
    
    # 打印第一个有标签和无标签样本的信息
    labeled_idx = next(i for i, x in enumerate(db_train.is_labeled) if x)
    unlabeled_idx = next(i for i, x in enumerate(db_train.is_labeled) if not x)

    print(f'labeled_index is {labeled_idx}')
    print(f'unlabeled_index is {unlabeled_idx}')
    
    print("\n有标签样本示例:")
    sample = db_train[labeled_idx]
    print("图像形状:", sample['image'].shape)
    print("标签形状:", sample['label'].shape)
    print("是否有标签:", sample['is_labeled'])
    
    print("\n无标签样本示例:")
    sample = db_train[unlabeled_idx]
    print("图像形状:", sample['image'].shape)
    print("标签形状:", sample['label'].shape)
    print("是否有标签:", sample['is_labeled'])