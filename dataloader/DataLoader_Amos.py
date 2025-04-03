import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 配置类，用于根据任务设置相关参数
class Config:
    def __init__(self, task):
        # 根据任务类型加载不同的数据路径和参数配置
        if task == "synapse":
            self.base_dir = './Datasets/Synapse'  # 基本数据目录
            self.save_dir = './synapse_data'  # 保存数据目录
            self.patch_size = (64, 128, 128)  # 图像的切片大小
            self.num_cls = 14  # 类别数
            self.num_channels = 1  # 输入图像的通道数
            self.n_filters = 32  # 网络中卷积层的滤波器数量
            self.early_stop_patience = 100  # 早停的耐心度（训练多少个周期没有提升就停止）
        else:  # 如果任务是 amos
            self.base_dir = './Datasets/amos22'  # 基本数据目录
            self.save_dir = './amos_data'  # 保存数据目录
            self.patch_size = (64, 128, 128)  # 图像的切片大小
            self.num_cls = 16  # 类别数
            self.num_channels = 1  # 输入图像的通道数
            self.n_filters = 32  # 网络中卷积层的滤波器数量
            self.early_stop_patience = 50  # 早停的耐心度（训练多少个周期没有提升就停止）

# 读取训练、验证或测试的样本列表
def read_list(split, task="synapse"):
    config = Config(task)  # 初始化配置
    ids_list = np.loadtxt(
        os.path.join(config.save_dir, 'splits', f'{split}.txt'),  # 从对应的路径加载split.txt文件
        dtype=str  # 强制数据为字符串类型
    ).tolist()
    return sorted(ids_list)  # 返回排序后的文件ID列表

# 读取数据：包括图像和标签
def read_data(data_id, task, nifti=False, test=False, normalize=False):
    config = Config(task)  # 初始化配置
    im_path = os.path.join(config.save_dir, 'npy', f'{data_id}_image.npy')  # 图像文件路径
    lb_path = os.path.join(config.save_dir, 'npy', f'{data_id}_label.npy')  # 标签文件路径

    # 如果路径不存在，抛出错误
    if not os.path.exists(im_path) or not os.path.exists(lb_path):
        raise ValueError(f"数据不存在：{data_id}")
    
    # 加载图像和标签数据
    image = np.load(im_path)
    label = np.load(lb_path)

    # 如果需要归一化，进行数据归一化处理
    if normalize:
        image = image.clip(min=-75, max=275)  # 限制图像的值域
        image = (image - image.min()) / (image.max() - image.min())  # 归一化到0-1区间
        image = image.astype(np.float32)  # 转换为32位浮点数

    return image, label  # 返回图像和标签

# 定义一个数据集类，继承自PyTorch的Dataset类
class Synapse_AMOS(Dataset):
    def __init__(self, split='train', repeat=None, transform=None, unlabeled=False, is_val=False, task="synapse", num_cls=1):
        # 初始化参数并加载数据
        self.ids_list = read_list(split, task=task)  # 加载指定split的ID列表
        self.repeat = repeat  # 设置数据重复次数
        self.task = task  # 任务类型
        if self.repeat is None:
            self.repeat = len(self.ids_list)  # 如果未指定重复次数，则使用数据集的长度
        print('total {} datas'.format(self.repeat))  # 输出数据集大小
        self.transform = transform  # 数据转换（例如数据增强）
        self.unlabeled = unlabeled  # 是否是无标签数据
        self.num_cls = num_cls  # 类别数
        self._weight = None  # 类别权重
        self.is_val = is_val  # 是否是验证集

        # 如果是验证集，直接将数据加载到内存中
        if self.is_val:
            self.data_list = {}
            for data_id in tqdm(self.ids_list):  # 遍历ID列表并加载数据
                image, label = read_data(data_id, task=task)
                self.data_list[data_id] = (image, label)

    # 返回数据集的长度
    def __len__(self):
        return self.repeat

    # 获取数据的方法
    def _get_data(self, data_id):
        if self.is_val:
            image, label = self.data_list[data_id]  # 如果是验证集，从内存中获取数据
        else:
            image, label = read_data(data_id, task=self.task)  # 否则从文件中读取数据
        return data_id, image, label

    # 获取一个样本
    def __getitem__(self, index):
        index = index % len(self.ids_list)  # 根据索引获取ID，确保不越界
        data_id = self.ids_list[index]  # 获取数据ID
        _, image, label = self._get_data(data_id)  # 获取对应的图像和标签

        if self.unlabeled:  # 如果是无标签数据，将标签置为0
            label[:] = 0

        # 归一化处理
        image = image.clip(min=-75, max=275)  # 限制像素值范围
        image = (image - image.min()) / (image.max() - image.min())  # 归一化到0-1

        sample = {'image': image, 'label': label}  # 包装样本

        if self.transform:
            sample = self.transform(sample)  # 如果有数据增强，进行转换

        return sample  # 返回样本

# 读取测试数据的函数
def read_test_data(num_samples=5, task="amos"):
    config = Config(task)  # 使用 AMOS 配置
    ids_list = read_list('train', task=task)  # 获取训练集列表
    
    # 获取前几个样本ID
    test_samples = ids_list[:num_samples]
    print(f"读取前{num_samples}个样本数据：")
    
    # 读取样本数据
    for data_id in test_samples:
        image, label = read_data(data_id, task=task, normalize=True)
        print(f"样本ID: {data_id}，图像形状: {image.shape}，标签形状: {label.shape}")
        # 这里可以进行其他测试操作，比如查看数据、可视化等

# 创建 DataLoader 用于训练或测试
def get_dataloader(dst_cls, args, split='train', repeat=None, unlabeled=False, config=None, transforms=None):
    dst = dst_cls(
        task=args.task,
        split=split,
        repeat=repeat,
        unlabeled=unlabeled,
        num_cls=config.num_cls,
        transform=transforms.Compose([
            RandomCrop(config.patch_size, args.task),
            RandomFlip_LR(),
            RandomFlip_UD(),
            ToTensor()
        ])
    )
    return DataLoader(
        dst,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker
    )

# 主程序入口
if __name__ == "__main__":
    # 读取并显示前5个样本的测试数据
    read_test_data(num_samples=5, task="amos")
