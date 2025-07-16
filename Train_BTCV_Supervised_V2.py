"""
适用于BTCV数据集的监督学习训练
使用DDP加快运算
"""

import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import tqdm
import os
import torch.nn.functional as F
import random
import json
import time
import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter # 启用Tensorboard
import logging # 日志系统
import argparse
from glob import glob

# BTCV
from dataloader.DataLoader_BTCV import BTCV

# 导入数据增强
from utils.ImageAugment import RandomRotFlip_LA as RandomRotFlip
from utils.ImageAugment import RandomCrop_LA as RandomCrop
from utils.ImageAugment import ToTensor_LA as ToTensor

# 导入DDP相关内容
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 导入网络框架
# from networks.VNet_MultiOutput_V2 import VNet
# from networks.VNet_MultiOutput_V3 import VNet
# from networks.VNet_MultiOutput_V4 import VNet
from networks.VNet_MultiOutput_V5 import VNet

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

# ==================== 主训练脚本 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP Training Script for 3D Medical Image Segmentation")

    # --- 基础训练参数 ---
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size PER GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    # --- 路径与日志 ---
    parser.add_argument("--log_path", type=str, default="./result_ddp/Training.log")
    parser.add_argument("--pth_path", type=str, default='./result_ddp/Pth')
    parser.add_argument("--tensorboard_path", type=str, default='./result_ddp/Train')

    # --- 控制参数 ---
    parser.add_argument("--continue_train", action="store_true", help="Resume from the latest checkpoint")

    # --- 数据集与模型 ---
    parser.add_argument("--dataset_path", type=str, default='./datasets/BTCV/data')
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--training_num", type=int, default=9)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    args = parser.parse_args()

    # ==================== DDP 环境初始化 ====================
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (local_rank <= 0)

    # ==================== 日志、种子、目录创建 ====================
    setup_logging(args.log_path)

    if is_main_process:
        logging.info(f"\n{'='*40} New DDP Training Session Started {'='*40}")
        logging.info(f"Using {world_size} GPUs for training.")
        logging.info(f"Training configuration:\n{json.dumps(vars(args), indent=2)}")
        os.makedirs(args.pth_path, exist_ok=True)
        os.makedirs(args.tensorboard_path, exist_ok=True)
        logging.info("Ensured model and TensorBoard directories exist.")

    set_seed(args.seed)
    logging.info(f"Process started on device: {device} with seed {args.seed}.")

    # ==================== TensorBoard (仅主进程) ====================
    writer = None
    if is_main_process:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        log_dir = os.path.join(args.tensorboard_path, timestamp)
        writer = SummaryWriter(log_dir=log_dir)
        logging.info(f"TensorBoard writer initialized at: {log_dir}")

    # ==================== 数据集和加载器 ====================
    logging.info("Initializing dataset...")
    list_path = './datasets/BTCV/btcv.txt'
    # 读取图像列表文件
    with open(list_path, 'r') as f:
        image_list = [line.strip() for line in f.readlines()]
        
    db_train = BTCV(
        base_dir='./datasets/BTCV/data',
        image_list=image_list,
        train_num=args.training_num,
        transform=transforms.Compose([RandomRotFlip(), RandomCrop(args.patch_size), ToTensor()])
    )
    logging.info(f"Dataset contains {len(db_train)} samples.")

    # [DDP 修改] 使用分布式采样器
    train_sampler = DistributedSampler(db_train, num_replicas=world_size, rank=local_rank, shuffle=True)

    train_loader = DataLoader(
        dataset=db_train,
        batch_size=args.batch_size,
        num_workers=max(1, os.cpu_count() // world_size), # 为每个进程分配合理的 worker
        pin_memory=True,
        sampler=train_sampler, # shuffle 必须为 False 或 None
        shuffle=False
    )
    logging.info(f"DataLoader created with DistributedSampler. Batch size per GPU: {args.batch_size}.")

    # ==================== 模型、损失函数、优化器 ====================
    logging.info("Initializing VNet model...")

    model = VNet(n_channels=1, n_classes=args.num_classes, normalization="batchnorm", has_dropout=True,n_filters=16).to(device) # VNet_V5
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0.01)

    start_epoch = 0
    if args.continue_train:
        logging.info("Attempting to continue training...")
        checkpoint_path = latest_checkpoint(args.pth_path)
        if checkpoint_path:
            start_epoch = load_model(model, checkpoint_path, device)
            logging.info(f"Resuming from checkpoint: {checkpoint_path}. Starting at epoch {start_epoch}.")
        else:
            logging.warning("`continue_train` is set, but no checkpoint found. Starting from scratch.")

    # [DDP 修改] 在加载权重后，用 DDP 包装模型
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    logging.info("Model wrapped with DDP. `find_unused_parameters` is set to True for multi-output model.")
        
    criterion = UnifiedLoss(num_classes=args.num_classes, lambda_reg=0.3, loss_weight=[1, 1])
    if is_main_process:
        logging.info(f"Loss function: {criterion.__class__.__name__} with λ_reg=0.3, weights=[1,1]")
        logging.info(f"Optimizer: {optimizer.__class__.__name__} with LR={args.learning_rate}")

    # ==================== 训练循环 ====================
    if is_main_process:
        logging.info(f"Starting training from epoch {start_epoch} to {args.epochs}...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_sampler.set_epoch(epoch) # [DDP 修改] 确保每个 epoch 的 shuffle 不同
        
        epoch_loss = 0.0
        
        # [DDP 修改] 只有主进程显示 TQDM 进度条
        pbar = None
        if is_main_process:
            pbar = tqdm.tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for i_batch, sampled_batch in enumerate(train_loader):
            images, labels = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
            
            optimizer.zero_grad()
            
            logits, variance, _ = model(images)
            total_loss = criterion(logits, variance, labels)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # [DDP 修改] 同步所有 GPU 的损失值以进行准确的日志记录
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
            epoch_loss += total_loss.item()
            
            if is_main_process:
                # 更新 TensorBoard
                global_step = epoch * len(train_loader) + i_batch
                writer.add_scalar('Loss/Batch_Total', total_loss.item(), global_step)
                writer.add_scalar('Uncertainty/Batch_Mean', variance.mean().item(), global_step)
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'AvgLoss': f'{epoch_loss/(i_batch+1):.4f}',
                    'Uncert': f'{variance.mean().item():.4f}'
                })
                pbar.update(1)
        
        if pbar: pbar.close()

        # [DDP 修改] 只有主进程记录 epoch 级别的日志和保存模型
        if is_main_process:
            avg_epoch_loss = epoch_loss / len(train_loader)
            logging.info(f"Epoch {epoch+1} finished. Average Training Loss: {avg_epoch_loss:.4f}")
            writer.add_scalar('Loss/Epoch_Avg', avg_epoch_loss, epoch + 1)
            
            if (epoch + 1) % 20 == 0 or (epoch + 1) == args.epochs:
                save_path = os.path.join(args.pth_path, f'model_epoch_{epoch+1}.pth')
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(), # [DDP 修改] 保存 .module 的状态
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_epoch_loss
                }
                torch.save(state, save_path)
                logging.info(f"Saved checkpoint for epoch {epoch+1} to {save_path}")

    # ==================== 训练结束 ====================
    if is_main_process:
        writer.close()
        logging.info("Training complete!")

    dist.destroy_process_group()
