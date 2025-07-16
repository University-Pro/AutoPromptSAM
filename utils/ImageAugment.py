"""
用于torchvision中的图像增强函数和类
"""
import torch
import numpy as np
import sys
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import rotate, zoom
from torch.utils.data.sampler import Sampler
import itertools
from skimage import transform as sk_trans
import random

# ————————————————————————————————————————————————————————————
# 来自MC-Net的数据增强类 for Amos
# ————————————————————————————————————————————————————————————

# 定义一个中心裁剪的类
class CenterCrop_Amos(object):
    def __init__(self, output_size, task):
        """
        初始化 CenterCrop 类
        Args:
        output_size (tuple): 裁剪后的目标尺寸
        task (str): 当前任务类型（例如 'synapse' 或 'amos'）
        """
        self.output_size = output_size  # 目标裁剪尺寸
        self.task = task  # 任务类型

    def __call__(self, sample):
        """
        裁剪图像并返回样本
        Args:
        sample (dict): 包含 'image' 和 'label' 的样本字典

        Returns:
        dict: 裁剪后的样本，包含裁剪后的图像和标签
        """
        image = sample['image']  # 获取图像
        # 判断图像是否需要补齐
        padding_flag = image.shape[0] <= self.output_size[0] or \
                       image.shape[1] <= self.output_size[1] or \
                       image.shape[2] <= self.output_size[2]

        # 如果图像尺寸小于目标尺寸，需要补齐
        if padding_flag:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)

        w1, h1, d1 = None, None, None
        ret_dict = {}

        # 计算裁剪的起始位置
        if w1 is None:
            (w, h, d) = image.shape
            w1 = int(round((w - self.output_size[0]) / 2.))
            if self.task == 'synapse':  # 对于 Synapse 数据集，按不同方式计算裁剪位置
                h1 = int(round((h // 2 - self.output_size[1]) / 2.))
                d1 = int(round((d // 2 - self.output_size[2]) / 2.))
            else:  # 对于其他任务
                h1 = int(round((h - self.output_size[1]) / 2.))
                d1 = int(round((d - self.output_size[2]) / 2.))

        # 遍历样本字典中的每个项目
        for key in sample.keys():
            item = sample[key]  # 获取当前键对应的数据
            if self.task == 'synapse':  # 对于 Synapse 数据集，可能需要对图像和标签进行调整
                dd, ww, hh = item.shape
                item = torch.FloatTensor(item).unsqueeze(0).unsqueeze(0)
                if key == 'image':  # 如果是图像，使用 trilinear 插值
                    item = F.interpolate(item, size=(dd, ww // 2, hh // 2), mode='trilinear', align_corners=False)
                else:  # 如果是标签，使用 nearest 插值
                    item = F.interpolate(item, size=(dd, ww // 2, hh // 2), mode="nearest")
                item = item.squeeze().numpy()
            # 如果图像需要补齐，进行补齐
            if padding_flag:
                item = np.pad(item, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            # 根据裁剪的起始位置进行裁剪
            item = item[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            ret_dict[key] = item  # 将处理后的数据存入字典

        return ret_dict  # 返回裁剪后的样本

# 定义一个随机裁剪的类
class RandomCrop_Amos(object):
    def __init__(self, output_size, task):
        """
        初始化 RandomCrop 类
        Args:
        output_size (tuple): 目标裁剪尺寸
        task (str): 当前任务类型（例如 'synapse' 或 'amos'）
        """
        self.output_size = output_size  # 目标裁剪尺寸
        self.task = task  # 任务类型

    def __call__(self, sample):
        """
        对图像进行随机裁剪
        Args:
        sample (dict): 包含 'image' 和 'label' 的样本字典

        Returns:
        dict: 裁剪后的样本，包含裁剪后的图像和标签
        """
        image = sample['image']  # 获取图像
        # 判断图像是否需要补齐
        padding_flag = image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]
        # 如果图像尺寸小于目标尺寸，需要补齐
        if padding_flag:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)

        w1, h1, d1 = None, None, None
        ret_dict = {}

        # 计算裁剪的随机起始位置
        if w1 is None:
            (w, h, d) = image.shape
            w1 = np.random.randint(0, w - self.output_size[0])
            if self.task == 'synapse':  # 对于 Synapse 数据集，按不同方式计算裁剪位置
                h1 = np.random.randint(0, h // 2 - self.output_size[1])
                d1 = np.random.randint(0, d // 2 - self.output_size[2])
            else:  # 对于其他任务
                h1 = np.random.randint(0, h - self.output_size[1])
                d1 = np.random.randint(0, d - self.output_size[2])

        # 遍历样本字典中的每个项目
        for key in sample.keys():
            item = sample[key]  # 获取当前键对应的数据
            if self.task == 'synapse':  # 对于 Synapse 数据集，可能需要对图像和标签进行调整
                dd, ww, hh = item.shape
                item = torch.FloatTensor(item).unsqueeze(0).unsqueeze(0)
                if key == 'image':  # 如果是图像，使用 trilinear 插值
                    item = F.interpolate(item, size=(dd, ww // 2, hh // 2), mode='trilinear', align_corners=False)
                else:  # 如果是标签，使用 nearest 插值
                    item = F.interpolate(item, size=(dd, ww // 2, hh // 2), mode="nearest")
                item = item.squeeze().numpy()
            # 如果图像需要补齐，进行补齐
            if padding_flag:
                item = np.pad(item, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            # 根据裁剪的随机起始位置进行裁剪
            item = item[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            ret_dict[key] = item  # 将处理后的数据存入字典

        return ret_dict  # 返回裁剪后的样本

# 定义一个水平翻转的类（左右翻转）
class RandomFlip_LR_Amos(object):
    def __init__(self, prob=0.5):
        """
        初始化水平翻转类
        Args:
        prob (float): 水平翻转的概率
        """
        self.prob = prob  # 翻转的概率

    def _flip(self, img, prob):
        """
        根据概率进行翻转
        Args:
        img (ndarray): 输入图像
        prob (tuple): 随机概率值

        Returns:
        ndarray: 翻转后的图像
        """
        if prob[0] <= self.prob:  # 如果随机值小于翻转概率，进行水平翻转
            img = np.flip(img, 1).copy()
        return img

    def __call__(self, sample):
        """
        对样本中的每个图像进行水平翻转
        Args:
        sample (dict): 包含 'image' 和 'label' 的样本字典

        Returns:
        dict: 包含翻转后图像和标签的样本字典
        """
        prob = (np.random.uniform(0, 1), np.random.uniform(0, 1))  # 随机生成水平和垂直翻转的概率
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]  # 获取当前键对应的数据
            item = self._flip(item, prob)  # 进行翻转
            ret_dict[key] = item  # 将翻转后的数据存入字典
        return ret_dict

# 定义一个垂直翻转的类（上下翻转）
class RandomFlip_UD_Amos(object):
    def __init__(self, prob=0.5):
        """
        初始化垂直翻转类
        Args:
        prob (float): 垂直翻转的概率
        """
        self.prob = prob  # 翻转的概率

    def _flip(self, img, prob):
        """
        根据概率进行翻转
        Args:
        img (ndarray): 输入图像
        prob (tuple): 随机概率值

        Returns:
        ndarray: 翻转后的图像
        """
        if prob[1] <= self.prob:  # 如果随机值小于翻转概率，进行垂直翻转
            img = np.flip(img, 2).copy()
        return img

    def __call__(self, sample):
        """
        对样本中的每个图像进行垂直翻转
        Args:
        sample (dict): 包含 'image' 和 'label' 的样本字典

        Returns:
        dict: 包含翻转后图像和标签的样本字典
        """
        prob = (np.random.uniform(0, 1), np.random.uniform(0, 1))  # 随机生成水平和垂直翻转的概率
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]  # 获取当前键对应的数据
            item = self._flip(item, prob)  # 进行翻转
            ret_dict[key] = item  # 将翻转后的数据存入字典
        return ret_dict

# 定义一个将数据转换为Tensor的类
class ToTensor_Amos(object):
    '''将样本中的ndarray类型转换为Tensor类型'''
    def __call__(self, sample):
        """
        将样本中的数据转换为Tensor
        Args:
        sample (dict): 包含 'image' 和 'label' 的样本字典

        Returns:
        dict: 转换为Tensor后的样本字典
        """
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            if key == 'image':  # 如果是图像
                ret_dict[key] = torch.from_numpy(item).unsqueeze(0).float()  # 转换为Float类型的Tensor
            elif key == 'label':  # 如果是标签
                ret_dict[key] = torch.from_numpy(item).long()  # 转换为Long类型的Tensor
            else:
                raise ValueError(key)
        
        return ret_dict  # 返回转换后的样本字典

# ————————————————————————————————————————————————————————————
# 来自MC-Net的数据增强类 for LA
# ————————————————————————————————————————————————————————————

# 随机旋转和翻转操作
def random_rot_flip_LA(image, label):
    # 随机选择旋转角度，旋转90度的倍数（0, 90, 180, 270度）
    k = np.random.randint(0, 4)
    # 对图像和标签进行旋转
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    # 随机选择翻转的轴（0: 垂直翻转, 1: 水平翻转）
    axis = np.random.randint(0, 2)
    # 对图像和标签进行翻转
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

# 随机旋转操作，角度范围为-20到20度
def random_rotate_LA(image, label):
    angle = np.random.randint(-20, 20)
    # 使用ndimage.rotate进行旋转，保持原始图像大小
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

# 数据增强类：随机旋转和翻转
class RandomGenerator_LA(object):
    def __init__(self, output_size):
        self.output_size = output_size  # 设置输出的图像尺寸

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # 随机进行旋转和翻转或者旋转操作
        if random.random() > 0.5:
            image, label = random_rot_flip_LA(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate_LA(image, label)
        
        # 获取图像的原始尺寸
        x, y = image.shape
        # 将图像和标签缩放到指定的输出尺寸
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # 转换为PyTorch张量
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # 变为四维张量: [1, C, H, W]
        label = torch.from_numpy(label.astype(np.uint8))  # 标签转换为无符号整数张量
        
        sample = {'image': image, 'label': label}
        return sample

# 数据增强类：调整图像大小
class Resize_LA(object):
    def __init__(self, output_size):
        self.output_size = output_size  # 设置输出尺寸

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # 获取图像的原始尺寸
        (w, h, d) = image.shape
        label = label.astype(np.bool)  # 标签转换为布尔类型
        # 使用skimage.transform.resize进行图像和标签的缩放
        image = sk_trans.resize(image, self.output_size, order=1, mode='constant', cval=0)
        label = sk_trans.resize(label, self.output_size, order=0)
        # 确保标签是二值化的
        assert(np.max(label) == 1 and np.min(label) == 0)
        assert(np.unique(label).shape[0] == 2)
        
        return {'image': image, 'label': label}

# 数据增强类：中心裁剪
class CenterCrop_LA(object):
    def __init__(self, output_size):
        self.output_size = output_size  # 设置输出的裁剪尺寸

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 如果标签尺寸小于目标尺寸，则进行填充
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            # 使用0值填充图像和标签
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        # 获取图像的原始尺寸
        (w, h, d) = image.shape

        # 计算裁剪区域的起始位置
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        # 根据计算的裁剪位置裁剪图像和标签
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}

# 数据增强类：随机裁剪
class RandomCrop_LA(object):
    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size  # 设置输出尺寸
        self.with_sdf = with_sdf  # 是否包括SDF（Signed Distance Function）

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # 如果标签尺寸小于目标尺寸，则进行填充
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            # 使用0值填充图像和标签
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        # 获取图像的原始尺寸
        (w, h, d) = image.shape

        # 随机选择裁剪区域的起始位置
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        # 裁剪图像和标签
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}

# 数据增强类：随机旋转和翻转
class RandomRotFlip_LA(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # 调用随机旋转和翻转函数
        image, label = random_rot_flip_LA(image, label)
        return {'image': image, 'label': label}

# 数据增强类：随机旋转
class RandomRot_LA(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # 调用随机旋转函数
        image, label = random_rotate_LA(image, label)
        return {'image': image, 'label': label}

# 数据增强类：添加噪声
class RandomNoise_LA(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu  # 噪声均值
        self.sigma = sigma  # 噪声标准差

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # 生成噪声并添加到图像上
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}

# 创建one-hot标签的类
class CreateOnehotLabel_LA(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes  # 类别数量

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # 创建one-hot标签
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}

# 转换为Tensor的类
class ToTensor_LA(object):
    """将ndarrays转换为PyTorch的Tensor"""
    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}

# 双流批量采样器
class TwoStreamBatchSampler_LA(Sampler):
    """遍历两个索引集
    'epoch'表示主索引集的一次遍历
    在epoch过程中，副索引集会遍历多次
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once_LA(self.primary_indices)
        secondary_iter = iterate_eternally_LA(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper_LA(primary_iter, self.primary_batch_size),
                    grouper_LA(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

class TwoStreamBatchSampler_LA_v2(torch.utils.data.sampler.Sampler):
    def __init__(self, labeled_idxs, unlabeled_idxs, batch_size, labeled_bs):
        self.labeled_idxs = np.array(labeled_idxs)
        self.unlabeled_idxs = np.array(unlabeled_idxs)
        self.batch_size = batch_size
        self.labeled_bs = labeled_bs
        self.unlabeled_bs = batch_size - labeled_bs
        
        # 计算完整遍历有标签数据所需的batch数
        self.num_batches = len(labeled_idxs) // labeled_bs
        
        # 扩展无标签数据索引以匹配有标签数据的batch需求
        self.unlabeled_idxs_cycled = np.tile(
            self.unlabeled_idxs,
            (self.num_batches * self.unlabeled_bs // len(unlabeled_idxs) + 1)
        )

    def __iter__(self):
        # 打乱有标签和无标签数据的顺序
        np.random.shuffle(self.labeled_idxs)
        np.random.shuffle(self.unlabeled_idxs_cycled)
        
        # 生成混合batch
        for i in range(self.num_batches):
            labeled_batch = self.labeled_idxs[i*self.labeled_bs : (i+1)*self.labeled_bs]
            unlabeled_batch = self.unlabeled_idxs_cycled[i*self.unlabeled_bs : (i+1)*self.unlabeled_bs]
            yield np.concatenate([labeled_batch, unlabeled_batch])

    def __len__(self):
        return self.num_batches  # 16//2=8 → 需强制改为16

# 生成一次迭代
def iterate_once_LA(iterable):
    return np.random.permutation(iterable)

# 生成无限次迭代
def iterate_eternally_LA(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

# 将数据分为固定长度的块
def grouper_LA(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)

# ————————————————————————————————————————————————————————————
# 来自Github的数据增强类，for LA，不进行任何更改
# ————————————————————————————————————————————————————————————
class RandomCrop_Github(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}
    
class CenterCrop_Github(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}

class RandomRotFlip_Github(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}

class ToTensor_Github(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}

class TwoStreamBatchSampler_Github(Sampler):
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
        primary_iter = iterate_once_Github(self.primary_indices)
        secondary_iter = iterate_eternally_Github(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper_Github(primary_iter, self.primary_batch_size),
                    grouper_Github(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once_Github(iterable):
    return np.random.permutation(iterable)

def iterate_eternally_Github(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper_Github(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)