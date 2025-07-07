import torch
import os
import sys
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import itertools
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
import h5py

class BTCV(Dataset):
    """ Synapse Dataset """
    def __init__(self, image_list, base_dir=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.image_list = image_list

        print("Total {} samples for training".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = self._base_dir + '/{}.h5'.format(image_name)
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class MACT(Dataset):
    """ Multi-organ Abdominal CT Reference Standard Segmentations Dataset """
    def __init__(self, image_list, base_dir=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.image_list = ['{:0>4}'.format(i + 1) for i in image_list]

        print("Total {} samples for training".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # ex: self._base_dir: '../data/MACT_h5'
        image_path = self._base_dir + '/{}.h5'.format(image_name)
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class RandomCrop(object):
    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, is_2d=False):
        self.is_2d = is_2d

    def __call__(self, sample):
        # image, label: For AHNet 2D to 3D,
        # 3D: WxHxD -> 1xWxHxD, 96x96x96 -> 1x96x96x96
        # 2D: WxHxD -> CxWxh, 224x224x3 -> 3x224x224
        image, label = sample['image'], sample['label']

        if self.is_2d:
            image = image.transpose(2, 0, 1).astype(np.float32)
            label = label.transpose(2, 0, 1)[1, :, :]
        else:
            # image = image.transpose(1, 0, 2)
            # label = label.transpose(1, 0, 2)
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)

        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long()}

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

def dataloader_test():
    # 配置数据集路径
    base_dir = "./datasets/BTCV/data"
    
    # 创建文件列表 (0001 到 0040)
    list_path = './datasets/BTCV/btcv.txt'
    
    # 读取图像列表文件
    with open(list_path, 'r') as f:
        image_list = [line.strip() for line in f.readlines()]

    # 创建数据集实例
    dataset = BTCV(image_list=image_list, base_dir=base_dir)
    
    print("\n===== 开始数据加载测试 =====")
    print(f"数据集大小：{len(dataset)} 个样本")
    
    # 测试随机样本
    for i in [0, 10, 20, 30, 39]:  # 测试不同位置的样本
        try:
            sample = dataset[i]
            img = sample['image']
            label = sample['label']
            
            print(f"\n样本 {i} ({image_list[i]}.h5):")
            print(f"  图像形状：{img.shape} | 数据类型：{img.dtype} | 值范围：[{img.min()}, {img.max()}]")
            print(f"  标签形状：{label.shape} | 数据类型：{label.dtype} | 唯一值：{np.unique(label)}")
            
        except Exception as e:
            print(f"加载样本 {i} 失败: {str(e)}")
    
    # 测试随机裁剪变换
    print("\n===== 测试数据变换 =====")
    crop_transform = RandomCrop(output_size=(128, 128, 128))
    sample = dataset[0]
    cropped = crop_transform(sample)
    
    print("随机裁剪 (128x128x128) 结果:")
    print(f"  原始图像形状: {sample['image'].shape}")
    print(f"  裁剪后形状: {cropped['image'].shape}")
    
    # 测试Tensor转换
    tensor_transform = ToTensor()
    tensor_sample = tensor_transform(cropped)
    
    print("\nTensor转换结果:")
    print(f"  图像Tensor形状: {tensor_sample['image'].shape} | 类型: {type(tensor_sample['image'])}")
    print(f"  标签Tensor形状: {tensor_sample['label'].shape} | 类型: {type(tensor_sample['label'])}")

if __name__ == "__main__":
    dataloader_test()