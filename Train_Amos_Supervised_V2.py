"""
适用于Amos数据集的监督学习代码
V2版本，相比于V1版本添加了DDP替换为原来的DP
增加了代码训练的速度，并保证了代码运行过程中的显存占用差别不大
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

# 导入DDP相关内容
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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

def latest_checkpoint(path):
    """在path中查找出最新的文件"""
    list_of_files = glob(os.path.join(path, '*.pth'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    if any(key.startswith('module.') for key in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    return model

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
    parser = argparse.ArgumentParser(description="Train a deep learning model for AMOS dataset")
    # 基础训练参数
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--log_path", type=str, default="./result/Training.log")
    parser.add_argument("--pth_path", type=str, default='./result/Pth')
    parser.add_argument("--tensorboard_path", type=str, default='./result/Train')
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--training_num", type=int, default=90)
    # 数据集与模型参数
    parser.add_argument("--dataset_path", type=str, default='./datasets/AMOS',
                        help="Base path for AMOS dataset")
    parser.add_argument("--num_classes", type=int, default=16,
                        help="Number of classes for AMOS (15 organs + background)")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[160,160,80],
                        help="Input patch size (xyz)")
    
    args = parser.parse_args()

    # ---------------- 日志和设备设置 ----------------
    setup_logging(args.log_path)
    logging.info(f"\n{'='*40} New Training Session Started {'='*40}")
        
    start_time = time.time()

    if args.multi_gpu:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f"cuda:{local_rank}")
        logging.info(f"Initialized distributed training on GPU {local_rank}. World size: {dist.get_world_size()}")
    else:
        local_rank = -1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using single device: {device}")

    logging.info(f"Training configuration:\n{json.dumps(vars(args), indent=2)}")

    # ---------------- Tensorboard 设置 ----------------
    writer = None
    if local_rank <= 0: # 仅主进程或单卡模式下初始化
        if args.continue_train and os.path.exists(args.tensorboard_path):
            log_dir = args.tensorboard_path
            logging.info(f"Resuming TensorBoard logging in existing directory: {log_dir}")
        else:
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            log_dir = os.path.join(args.tensorboard_path, timestamp)
            logging.info(f"Creating new TensorBoard log directory: {log_dir}")
        writer = SummaryWriter(log_dir=log_dir)

    # ---------------- 数据集和加载器 ----------------
    logging.info("Initializing AMOS dataset...")
    db_train = AmosDataset(
        split='train',
        training_num=args.training_num,
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomCrop(args.patch_size),
            ToTensor()
        ]),
        preload=True
    )
    logging.info(f"Successfully loaded {len(db_train)} training samples.")

    train_sampler = DistributedSampler(db_train) if args.multi_gpu else None
    train_loader = DataLoader(
        dataset=db_train,
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
        sampler=train_sampler
    )
    logging.info(f"DataLoader created with batch size: {args.batch_size}, num_workers: {os.cpu_count()}, shuffle: {train_sampler is None}.")

    # ---------------- 模型、损失函数、优化器 ----------------
    logging.info("Initializing VNet model...")
    model = VNet(n_channels=1, n_classes=args.num_classes, normalization="batchnorm", has_dropout=True)
    logging.info(f"Model VNet created with {args.num_classes} classes and batch normalization.")
    
    model.to(device)

    start_epoch = 0
    if args.continue_train:
        logging.info("Attempting to continue training...")
        checkpoint = latest_checkpoint(args.pth_path)
        if checkpoint:
            start_epoch, loss = load_model(model, checkpoint, device)
            logging.info(f"Successfully loaded checkpoint: {checkpoint}. Resuming from epoch {start_epoch + 1}. Last recorded loss: {loss:.4f}")
        else:
            logging.warning("`continue_train` is set, but no checkpoint was found. Starting from scratch.")
    else:
        logging.info("Starting training from scratch.")

    if args.multi_gpu:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) # find_unused_parameters=True 在某些模型下是必要的
        logging.info("Wrapped model with DistributedDataParallel (DDP).")
        
    criterion = UnifiedLoss(num_classes=args.num_classes, lambda_reg=0.3, loss_weight=[1, 1])
    logging.info(f"Using loss function: {criterion.__class__.__name__} with parameters: num_classes={args.num_classes}, lambda_reg=0.3, loss_weight=[1,1]")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0.01)
    logging.info(f"Optimizer: {optimizer.__class__.__name__} with learning rate: {args.learning_rate}, weight_decay: 0.01")
    
    # ---------------- 训练循环 ----------------
    logging.info(f"Starting model training from epoch {start_epoch + 1} to {args.epochs}...")
    model.train()
    
    for epoch in range(start_epoch, args.epochs):
        if args.multi_gpu:
            train_sampler.set_epoch(epoch)
            
        epoch_loss = 0.0 # [修改] 用于累计整个epoch的loss
        # [修改] 优化tqdm描述
        pbar_desc = f'Epoch {epoch+1}/{args.epochs}'
        if local_rank > 0: # 非主进程不显示进度条
             pbar = iter(train_loader)
        else:
             pbar = tqdm.tqdm(train_loader, desc=pbar_desc, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        for i_batch, sampled_batch in enumerate(pbar if local_rank <= 0 else train_loader):
            images, labels = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
            
            optimizer.zero_grad()
            
            logits, variance, features = model(images)
            total_loss = criterion(logits, variance, labels.squeeze(1))
            
            # 在DDP中，损失已经是每个GPU上的均值， backward前不需要额外处理
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item() # [修改] 累加loss
            
            if local_rank <= 0: # 仅主进程更新TensorBoard和进度条
                global_step = epoch * len(train_loader) + i_batch
                writer.add_scalar('Loss/Batch_Total', total_loss.item(), global_step)
                writer.add_scalar('Uncertainty/Batch_Mean', variance.mean().item(), global_step)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'AvgLoss': f'{epoch_loss/(i_batch+1):.4f}',
                    'Uncert': f'{variance.mean().item():.4f}'
                })
        
        # 日志记录Loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        if local_rank <= 0: # 仅主进程记录epoch日志和保存模型
            logging.info(f"Epoch {epoch+1}/{args.epochs} finished. Average Training Loss: {avg_epoch_loss:.4f}")
            writer.add_scalar('Loss/Epoch_Avg', avg_epoch_loss, epoch + 1)
            
            if (epoch + 1) % 20 == 0 or (epoch + 1) == args.epochs: 
                save_dir = args.pth_path
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}_checkpoint.pth')
                torch.save(model.module.state_dict() if args.multi_gpu else model.state_dict(), save_path)
                logging.info(f"Saved checkpoint for epoch {epoch+1} to {save_path}")

    # ---------------- 训练结束 ----------------
    if local_rank <= 0:
        if writer:
            writer.close()
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        # [新增日志] 记录总耗时
        logging.info(f"Training complete! Total duration: {hours}h {minutes}m {seconds}s")
        logging.info(f"{'='*40} Training Session Ended {'='*40}\n")
        
    if args.multi_gpu:
        dist.destroy_process_group()
