"""
适用于Amos数据集的监督学习代码
可以设置有标签的数量
"""

import argparse
import os
import glob
import logging
from pyexpat import features
import time
import numpy as np
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter # 启用Tensorboard
import logging # 日志系统
from glob import glob

# Amos数据集的Dataloader
from dataloader.DataLoader_Amos import AmosDataset
from dataloader.DataLoader_Amos import AmosConfig

# 导入数据增强
from utils.ImageAugment import RandomRotFlip_LA as RandomRotFlip
from utils.ImageAugment import RandomCrop_LA as RandomCrop
from utils.ImageAugment import ToTensor_LA as ToTensor

# 导入网络框架
from networks.VNet_MultiOutput_V2 import VNet

# 导入Loss函数
from utils.LA_Train_Metrics import softmax_mse_loss
from utils.LA_Train_Metrics import kl_loss
from utils.LA_Train_Metrics import CeDiceLoss
from utils.LA_Train_Metrics import nDiceLoss

def set_seed(seed_value=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def setup_logging(log_file):
    """日志记录"""
    log_dir = os.path.dirname(log_file)
    # 如果日志文件夹没有，就创建文件夹
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(log_file, mode='a'),
                                  logging.StreamHandler()])
    
    # 测试日志记录器是否正常工作
    logging.info("Logging is set up.")

# 查找最新的文件
def latest_checkpoint(path):
    """在path中查找出最新的文件"""
    list_of_files = glob(os.path.join(path, '*.pth'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# 自动处理多卡和单卡模型
def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # If loading a DataParallel model, remove `module.` prefix
    if any(key.startswith('module.') for key in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v  # remove `module.`
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    return model

class UnifiedLoss_V1(nn.Module):
    def __init__(self, lambda_reg=0.5):
        super().__init__()
        self.lambda_reg = lambda_reg  # 正则化项权重

    def uncertainty_regularizer(self, ce_loss_per_pixel, variance):
        variance = variance.squeeze(1)  # [B,H,W,D]
        eps = 1e-6
        variance = variance + eps
        reg_term = (0.5 * ce_loss_per_pixel / variance) + 0.5 * torch.log(variance)
        return reg_term.mean()

    def forward(self, logits, variance, target):
        ce_per_pixel = nn.CrossEntropyLoss(reduction='none')(logits, target)
        seg_loss = ce_per_pixel.mean()
        reg_loss = self.uncertainty_regularizer(ce_per_pixel.detach(), variance)
        return seg_loss + self.lambda_reg * reg_loss

class UnifiedLoss(nn.Module):
    def __init__(self, num_classes, lambda_reg=0.5, loss_weight=[0.4, 0.6]):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_reg = lambda_reg  # 不确定性正则化权重
        self.loss_weight = loss_weight  # [CE权重, Dice权重]
        
        # 初始化基础损失组件
        self.celoss = nn.CrossEntropyLoss(reduction='none')
        self.diceloss = nDiceLoss(num_classes)
        self.softplus = nn.Softplus()

    def uncertainty_regularizer(self, ce_loss_per_pixel, variance):
        """计算基于交叉熵的不确定性正则项"""
        variance = variance.squeeze(1)  # [B,H,W,D]
        eps = 1e-6
        variance = variance.clamp(min=eps)  # 数值稳定性
        
        # 组合正则项
        reg_term = (0.5 * ce_loss_per_pixel.detach() / variance) + 0.5 * torch.log(variance)
        return reg_term.mean()

    def forward(self, logits, variance, target):
        """
        参数：
            logits: 网络输出的logits [B, C, H, W, D]
            variance: 不确定性头输出 [B, 1, H, W, D]
            target: 标签 [B, H, W, D] 或 [B, 1, H, W, D]
        """
        # 维度处理
        if target.dim() == 5:
            target = target.squeeze(1)  # 压缩通道维度
        target = target.long()  # 确保为整数类型

        # 计算逐像素交叉熵（用于正则项）
        ce_per_pixel = self.celoss(logits, target)  # [B, H, W, D]
        ce_loss = ce_per_pixel.mean()  # 平均CE损失

        # 计算Dice损失
        dice_loss = self.diceloss(logits, target, softmax=True)  # 假设diceloss内部处理softmax

        # 组合基础分割损失
        seg_loss = self.loss_weight[0] * ce_loss + self.loss_weight[1] * dice_loss

        # 计算不确定性正则项
        reg_loss = self.uncertainty_regularizer(ce_per_pixel, variance)

        # 总损失 = 分割损失 + 正则项
        total_loss = seg_loss + self.lambda_reg * reg_loss

        return total_loss

if __name__ == "__main__":
    # 参数解析器增强
    parser = argparse.ArgumentParser(description="Train a deep learning model for AMOS dataset")
    # 基础训练参数
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)  # 已修改
    parser.add_argument("--log_path", type=str, default="./result/Training.log")
    parser.add_argument("--pth_path", type=str, default='./result/Pth')
    parser.add_argument("--tensorboard_path", type=str, default='./result/Train')
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--training_num", type=int, default=90)  # 已修改
    
    # 数据集与模型参数 (专为AMOS定制)
    parser.add_argument("--dataset_path", type=str, default='./datasets/AMOS',
                       help="Base path for AMOS dataset")
    parser.add_argument("--num_classes", type=int, default=16, # 已修改
                       help="Number of classes for AMOS (15 organs + background)")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[80, 160, 160],  # 已修改
                       help="Input patch size (z, y, x)")
    args = parser.parse_args()

    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # 设置日志
    setup_logging(args.log_path)
    logging.info(f"\n{'='*40} New Training Session {'='*40}")
    logging.info(f"Configuration:\n{json.dumps(vars(args), indent=4)}")

    # 设置Tensorboard
    if args.continue_train and os.path.exists(args.tensorboard_path):
        log_dir = args.tensorboard_path
    else:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        log_dir = os.path.join(args.tensorboard_path, timestamp)
    logging.info(f"TensorBoard logging directory: {log_dir}")

    # 设置TensorBoard记录器
    writer = SummaryWriter(log_dir=log_dir)

    # 创建数据集
    logging.info("Initializing AMOS dataset...")
    db_train = AmosDataset(
        split='train',
        training_num=args.training_num,
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomCrop(args.patch_size),
            ToTensor()
        ]),
        preload=True  # AMOS推荐预加载
    )
    logging.info(f"Successfully loaded {len(db_train)} training samples from AMOS dataset.")
    
    # 加载数据
    train_loader = DataLoader(
        dataset=db_train,
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),  # 自动适配CPU核心数
        pin_memory=True,
        shuffle=True,
        persistent_workers=True  # 保持worker进程
    )

    # 定义模型
    logging.info("Initializing VNet model...")
    model = VNet(n_channels=1, n_classes=args.num_classes, normalization="batchnorm", has_dropout=True) # VNet
    logging.info(f"Model created with {args.num_classes} classes.")

    # 定义损失函数
    criterion = UnifiedLoss(num_classes=args.num_classes, lambda_reg=0.3, loss_weight=[1, 1])
    logging.info(f"Using Unified Uncertainty-aware Loss with num_classes={args.num_classes}, lambda_reg=0.3, loss_weight=[1,1")

    # 如果继续训练
    if args.continue_train:
        checkpoint = latest_checkpoint(args.pth_path)
        if checkpoint:
            load_model(model, checkpoint, "cuda")

    # 多卡模型
    if args.multi_gpu:
        model = nn.DataParallel(model)

    model.to(device)

    # 设置优化器
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.01) # 优化器
    logging.info(f"Optimizer: {optimizer.__class__.__name__} with parameters: {optimizer.defaults}")

    # 模型设置为训练模式
    model.train()

    logging.info("Starting model training...")
    for epoch in range(args.epochs):
        running_loss = 0.0

        pbar = tqdm.tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{args.epochs}', bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        for i_batch, sampled_batch in enumerate(train_loader):
            images, labels = sampled_batch['image'].to(device), sampled_batch['label'].to(device)  # 加载数据到设备
            
            optimizer.zero_grad()

            # 全精度训练
            logits, variance, features = model(images)
            # 计算总损失（包含不确定性正则项）
            total_loss = criterion(logits, variance, labels.squeeze(1))

            # 聚合所有GPU的loss
            if args.multi_gpu:
                total_loss = total_loss.mean()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 增加梯度裁剪
            optimizer.step()

            # 更新监控指标
            running_loss += total_loss.item()

            # 记录到TensorBoard（添加不确定性统计）
            global_step = epoch * len(train_loader) + i_batch
            writer.add_scalar('Loss/Total', total_loss.item(), global_step)
            writer.add_scalar('Uncertainty/Mean', variance.mean().item(), global_step)
            writer.add_scalar('Uncertainty/Max', variance.max().item(), global_step)

            # 更新进度条显示更多信息
            pbar.set_postfix({
                'Total Loss': f'{running_loss/(i_batch+1):.4f}',
                'Uncert': f'{variance.mean().item():.4f}'  # 显示平均不确定性
            })

            # 日志输出当前的Loss
            pbar.update(1)

        pbar.close()

        # 日志记录当前的Loss
        logging.info(f"Epoch {epoch+1}/{args.epochs}, Batch {i_batch+1}/{len(train_loader)}, Loss: {total_loss.item():.4f}")

        # 保存模型
        if (epoch + 1) % 20 == 0:
            # 检查目录是否存在，如果不存在则创建
            if not os.path.exists(args.pth_path):
                os.makedirs(args.pth_path)
                logging.info(f"Created directory: {args.pth_path}")
            
            # 保存模型
            temp_path = os.path.join(args.pth_path, f'model_epoch_{epoch+1}_checkpoint.pth')
            torch.save(model.module.state_dict() if args.multi_gpu else model.state_dict(), temp_path)
            logging.info(f"Saved checkpoint at epoch {epoch+1} at {temp_path}")

    writer.close()
    logging.info("Training complete!")