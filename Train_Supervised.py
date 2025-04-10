"""
重写训练代码
将训练逻辑写入到Main函数中
其他的放到函数中
用于监督学习
"""

import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import tqdm
import os
import torch.nn.functional as F
import random
import time
import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter # 启用Tensorboard
import logging # 日志系统
import argparse
from glob import glob

# 完整版本的Synapse的Dataloader
from dataloader.DataLoader_Synapse import Synapse_dataset
from dataloader.DataLoader_Synapse import RandomGenerator

# LA的DataLoader
# from dataloader.DataLoader_LA import LAHeart
# from dataloader.DataLoader_LA_Semi import LAHeart

# Synapse的Dataloader
from dataloader.DataLoader_Synapse import Synapse_dataset

# 导入LA的数据增强
from utils.ImageAugment import RandomRotFlip_LA as RandomRotFlip
from utils.ImageAugment import RandomCrop_LA as RandomCrop
from utils.ImageAugment import ToTensor_LA as ToTensor

# 导入加载无标签的工具
from utils.ImageAugment import TwoStreamBatchSampler_LA

# 导入网络框架
from networks.VNet import VNet

# 导入Loss函数
from utils.LA_Train_Metrics import softmax_mse_loss
from utils.LA_Train_Metrics import CeDiceLoss
from utils.LA_Train_Metrics import Github_Loss

def set_seed(seed_value=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

if __name__ == "__main__":
    # 参数解析器
    parser = argparse.ArgumentParser(description="Train a deep learning model on a given dataset")
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument('--dataset_path', type=str, default='./datasets/LA')
    parser.add_argument("--log_path", type=str, default="./result/Training.log")
    parser.add_argument("--pth_path", type=str, default='./result/Pth')
    parser.add_argument("--tensorboard_path", type=str, default='./result/Train')
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--training_num", type=int, default=80)
    parser.add_argument("--use_amp", action="store_true", help="Whether to use mixed precision training") # 如果使用AMP就加上这个参数

    args = parser.parse_args()

    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler() if args.use_amp else None  # 如果启用混合精度，则创建 GradScaler

    # 设置日志
    setup_logging(args.log_path)

    # 设置Tensorboard
    if args.continue_train and os.path.exists(args.tensorboard_path):
        log_dir = args.tensorboard_path
    else:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        log_dir = os.path.join(args.tensorboard_path, timestamp)
    logging.info(f"TensorBoard logging directory: {log_dir}")

    # 设置TensorBoard记录器
    writer = SummaryWriter(log_dir=log_dir)

    # 创建LA数据集
    patch_size = (112, 112, 80)
    # db_train = LAHeart(base_dir=args.dataset_path,
    #                    split='train_label',
    #                    transform=transforms.Compose([
    #                         RandomRotFlip(),
    #                         RandomCrop(patch_size),
    #                         ToTensor()
    #                         # transforms.Normalize(mean=[0.5], std=[0.5]) # 添加数据归一化
    #                     ]),
    #                     num=args.training_num)
    
    # 创建Synapse数据集
    db_train = Synapse_dataset(base_dir="./datasets/Synapse/data",
                               list_dir="./datasets/Synapse/list",
                               split="train",
                               transform = RandomGenerator((224,224)))

    logging.info(f"Training dataset: {len(db_train)}")
    
    # 加载数据
    train_loader = DataLoader(
        dataset=db_train,
        batch_size=args.batch_size,
        num_workers=8,  # 自动适配CPU核心数
        pin_memory=True,
        shuffle=True,
        persistent_workers=True  # 保持worker进程
    )

    # 定义模型
    model = VNet(n_channels=1, n_classes=args.num_classes, normalization='batchnorm', has_dropout=True).to(device=device)

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

    # 定义损失函数
    criterion_dice = CeDiceLoss(num_classes=args.num_classes, loss_weight=[0.6, 1.4])
    logging.info(f"Using CeDiceLoss")

    # 学习率调度器
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=50,         # 初始周期长度
    #     T_mult=2,       # 周期倍增系数
    #     eta_min=1e-6
    # )

    # 设置新的学习率调度器
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=1e-3,           # 与优化器初始学习率一致
    #     epochs=args.epochs,
    #     steps_per_epoch=len(train_loader),
    #     pct_start=0.1,         # 10%的预热阶段
    #     anneal_strategy='cos'
    # )

    # 模型设置为训练模式
    model.train()

    for epoch in range(args.epochs):
        running_loss = 0.0

        pbar = tqdm.tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{args.epochs}', bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        for i_batch, sampled_batch in enumerate(train_loader):
            images, labels = sampled_batch['image'].to(device), sampled_batch['label'].to(device) # 原先内容
            
            optimizer.zero_grad()

            # 使用混合精度
            if args.use_amp:
                with torch.amp.autocast():  # 启用混合精度计算
                    # 模型输出
                    sam2_output = model(images)

                    # **监督损失**: Dice Loss between sam2_output and ground truth
                    loss_dice = criterion_dice(sam2_output, labels)

                    # 总损失
                    total_loss = loss_dice
            else:
                # 全精度训练
                sam2_output = model(images) # 原先的坂本
                loss_dice = criterion_dice(sam2_output, labels)
                total_loss = loss_dice

            # 混合精度训练下的反向传播
            if args.use_amp:
                scaler.scale(total_loss).backward()  # 缩放梯度
                scaler.unscale_(optimizer)  # 解缩放
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 增加梯度裁剪
                optimizer.step()

            # 更新损失记录
            running_loss += total_loss.item()

            # 记录到TensorBoard
            global_step = epoch * len(train_loader) + i_batch
            writer.add_scalar('Loss/Total', total_loss.item(), global_step)
            writer.add_scalar('Loss/Dice', loss_dice.item(), global_step)

            # 更新进度条
            pbar.set_postfix({
                'Total Loss': f'{running_loss/(i_batch+1):.4f}',
            })

            # 日志输出当前的Loss
            pbar.update(1)


        pbar.close()
        # scheduler.step() # 不设置学习率调度器

        # 日志记录当前的Loss
        logging.info(f"Epoch {epoch+1}/{args.epochs}, Batch {i_batch+1}/{len(train_loader)}, Loss: {total_loss.item():.4f}")

        # 保存模型
        if (epoch + 1) % 10 == 0:
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