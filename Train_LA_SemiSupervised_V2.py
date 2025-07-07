"""
适用于利用LA数据集进行半监督学习的相关代码
相比于V1添加了DDP训练，提高了训练效率和显卡占用率
这里的DDP需要修改双流编码器
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
from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk
from torch.optim.lr_scheduler import StepLR  # 动态学习率
from torch.utils.tensorboard import SummaryWriter  # 启用Tensorboard
import logging  # 日志系统
import argparse
from glob import glob
import json

# LA的DataLoader
from dataloader.DataLoader_LA import LAHeart

# 导入LA的数据增强
from utils.ImageAugment import RandomRotFlip_LA as RandomRotFlip
from utils.ImageAugment import RandomCrop_LA as RandomCrop
from utils.ImageAugment import ToTensor_LA as ToTensor

# 导入加载无标签的工具
from utils.ImageAugment import TwoStreamBatchSampler_LA

# 导入DDP相关内容
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 导入网络框架
from networks.SAM3D_VNet_SSL_V15 import Network

# 导入Loss函数
from utils.LA_Train_Metrics import softmax_mse_loss
from utils.LA_Train_Metrics import kl_loss
from utils.LA_Train_Metrics import CeDiceLoss

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

def latest_checkpoint(path):
    """在path中查找出最新的文件"""
    list_of_files = glob(os.path.join(path, '*.pth'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def load_model(model, model_state_dict, device):
    """
    通用的加载模型函数，主要处理双卡前缀问题
    """
    # 判断当前模型是否是 DataParallel 实例
    is_dataparallel = isinstance(model, nn.DataParallel)
    # 动态处理键名前缀
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        # 情况1：当前是 DataParallel，但检查点没有 'module.' 前缀 → 添加前缀
        if is_dataparallel and not k.startswith('module.'):
            new_key = 'module.' + k
        # 情况2：当前不是 DataParallel，但检查点有多余的 'module.' 前缀 → 移除前缀
        elif not is_dataparallel and k.startswith('module.'):
            new_key = k[7:]  # 移除 'module.'
        # 其他情况：保持原样
        else:
            new_key = k
        new_state_dict[new_key] = v
    # 加载调整后的权重
    model.load_state_dict(new_state_dict)
    return model

def load_pretrained_v2(model, pretrained_path,multi_gpu=False):
    """
    改进版权重加载函数，支持自动处理多GPU前缀和分支前缀
    并且支持非严格加载
    """
    pretrained_dict = torch.load(pretrained_path)
    
    print("\n===== 原始权重键名样例 =====")
    print(list(pretrained_dict.keys())[:5])
    
    # 第一步：去除预训练权重中的module前缀（如果存在）
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    
    # 第二步：构建目标键名映射
    model_dict = model.state_dict()
    updated_dict = {}
    mismatch_keys = []
    
    # 根据是否多GPU决定目标前缀
    target_prefix = "module.unet_branch1." if multi_gpu else "unet_branch1."
    
    print("\n===== 模型期待键名样例 =====")
    print([k for k in model_dict.keys() if "unet_branch1" in k][:5])
    
    for k, v in pretrained_dict.items():
        # 情况1：权重来自标准UNet（无分支前缀）
        if not k.startswith(("unet_branch1.", "unet_branch2.")):
            # 尝试直接添加分支前缀
            new_key = target_prefix + k
            if new_key in model_dict and model_dict[new_key].shape == v.shape:
                updated_dict[new_key] = v
                continue
            
            # 尝试匹配子模块（如inc.double_conv）
            for module_part in ["inc.", "down", "up", "outc"]:
                if module_part in k:
                    new_key = target_prefix + module_part + k.split(module_part)[1]
                    if new_key in model_dict and model_dict[new_key].shape == v.shape:
                        updated_dict[new_key] = v
                        break
        
        # 情况2：权重已有部分匹配前缀
        elif k.startswith("unet_branch1."):
            new_key = target_prefix + k.replace("unet_branch1.", "")
            if new_key in model_dict:
                updated_dict[new_key] = v
                continue
        
        # 记录不匹配的键
        if k not in updated_dict:
            mismatch_keys.append(k)

    # 打印调试信息
    if mismatch_keys:
        print("\n===== 未加载的权重键 =====")
        print(f"总数: {len(mismatch_keys)}/{len(pretrained_dict)}")
        print("示例:", mismatch_keys[:5])
    
    # 安全加载并打印统计
    model.load_state_dict(updated_dict, strict=False)
    
    # 计算加载成功率
    loaded_keys = set(updated_dict.keys())
    expected_keys = set(k for k in model_dict.keys() if "unet_branch1" in k)
    
    print("\n===== 加载结果统计 =====")
    print(f"成功加载: {len(loaded_keys)}/{len(expected_keys)} 参数")
    print(f"匹配率: {len(loaded_keys)/len(expected_keys):.1%}")
    
    return model

class DistributedTwoStreamBatchSampler(TwoStreamBatchSampler_LA):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size,
                 num_replicas=None, rank=None, seed=21):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
            
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.batch_size = batch_size
        self.secondary_batch_size = secondary_batch_size
        
        # 计算每个进程应该处理的大约批次数量
        self.primary_batch_size = self.batch_size - self.secondary_batch_size
        
        # 确保有标签和无标签数据的数量足够
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

        self.num_primary_batches = len(self.primary_indices) // self.primary_batch_size
        self.num_secondary_batches_per_primary = len(self.secondary_indices) // self.secondary_batch_size
        
        # 以有标签数据为主导，计算总批次数
        self.total_batches = self.num_primary_batches
        self.num_samples = self.total_batches // self.num_replicas
        
        logging.info(f"[Rank {self.rank}] Initialized DistributedTwoStreamBatchSampler.")
        logging.info(f"[Rank {self.rank}] Total batches per epoch: {self.total_batches}, Samples per replica: {self.num_samples}")

    def __iter__(self):
        # 1. 创建随机数生成器，并使用 epoch + seed 来保证每个 epoch 的 shuffle 不同
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # 2. 对有标签和无标签的索引进行 shuffle
        primary_iter = torch.randperm(len(self.primary_indices), generator=g).tolist()
        secondary_iter = torch.randperm(len(self.secondary_indices), generator=g).tolist()

        # 3. 生成完整的批次列表
        batches = []
        for i in range(self.total_batches):
            # 取有标签数据索引
            primary_start_idx = i * self.primary_batch_size
            primary_batch_indices = [self.primary_indices[j] for j in primary_iter[primary_start_idx : primary_start_idx + self.primary_batch_size]]
            
            # 取无标签数据索引 (循环使用)
            secondary_start_idx = (i * self.secondary_batch_size) % len(secondary_iter)
            secondary_end_idx = secondary_start_idx + self.secondary_batch_size
            
            if secondary_end_idx > len(secondary_iter):
                # 如果超出了范围，就从头取
                secondary_batch_indices_raw = secondary_iter[secondary_start_idx:] + secondary_iter[:secondary_end_idx - len(secondary_iter)]
            else:
                secondary_batch_indices_raw = secondary_iter[secondary_start_idx:secondary_end_idx]

            secondary_batch_indices = [self.secondary_indices[j] for j in secondary_batch_indices_raw]
            
            # 合并批次
            batch = primary_batch_indices + secondary_batch_indices
            batches.append(batch)
            
        # 4. 根据 rank 分发批次
        # 这是分布式采样的关键：每个进程只获取自己的一部分批次
        indices = batches[self.rank:len(batches):self.num_replicas]
        logging.info(f"[Rank {self.rank}, Epoch {self.epoch}] Yielding {len(indices)} batches.")
        
        return iter(indices)

    def __len__(self):
        # 返回当前进程需要处理的批次数
        return (self.total_batches + self.num_replicas - 1) // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch
        logging.info(f"[Rank {self.rank}] Sampler epoch set to {self.epoch}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="3D医学图像分割训练脚本（半监督-DDP）")
    # --- 训练参数 ---
    parser.add_argument('--seed', type=int, default=21, help='随机种子')
    parser.add_argument("--epochs", type=int, default=200, help='训练总轮数')
    parser.add_argument("--batch_size", type=int, default=4, help='每个GPU的总批量大小（有+无标签）')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help='初始学习率')
    # --- 路径参数 ---
    parser.add_argument('--dataset_path', type=str, default='./datasets/LA', help='数据集路径')
    parser.add_argument("--log_path", type=str, default="./result/Training.log", help='日志文件路径')
    parser.add_argument("--pth_path", type=str, default='./result/SAM3D_VNet_SSL/LA/Pth', help='模型保存路径')
    parser.add_argument("--tensorboard_path", type=str, default='./result/Train', help='TensorBoard日志路径')
    # --- 控制参数 ---
    parser.add_argument("--continue_train", action="store_true", help='是否继续训练')
    # --- 半监督参数 ---
    parser.add_argument("--training_label_num", type=int, default=16, help='有标签样本数量')
    parser.add_argument("--training_unlabel_num", type=int, default=64, help='无标签样本数量')
    parser.add_argument("--label_bs", type=int, default=2, help='每个GPU的有标签样本批量大小')
    parser.add_argument('--num_classes', type=int, default=2, help='类别数量')
    parser.add_argument("--consistency_weight", type=float, default=0.1, help='一致性损失权重') 
    # --- 预训练加载 ---
    parser.add_argument("--pretrained_weights", type=str, default=None, help='预训练权重路径')
    args = parser.parse_args()
    
    # ==================== DDP环境初始化 ====================
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    is_main_process = (local_rank <= 0)

    # ==================== 文件、日志、随机种子 ====================
    if is_main_process:
        for path in [os.path.dirname(args.log_path), args.pth_path, args.tensorboard_path]:
            os.makedirs(path, exist_ok=True)
        setup_logging(args.log_path, local_rank)
        logging.info(f"\n{'='*30} New DDP Training Session Started {'='*30}")
        logging.info(f"Distributed training with {world_size} GPUs.")
        logging.info(f"Training configuration:\n{json.dumps(vars(args), indent=2)}")
    else:
        # 非主进程也设置日志，但只输出到控制台
        setup_logging(args.log_path, local_rank)
    
    set_seed(args.seed)
    logging.info(f"[Rank {local_rank}] Process started on device: {device}")
    
    # ==================== 数据准备 ====================
    logging.info(f"[Rank {local_rank}] Preparing dataset and dataloader...")
    patch_size = (112, 112, 80)
    train_transform = transforms.Compose([
        RandomRotFlip(),
        RandomCrop(patch_size),
        ToTensor()
    ])
    full_dataset = LAHeart(base_dir=args.dataset_path, split='train', transform=train_transform)
    
    labeled_idx = list(range(args.training_label_num))
    unlabeled_idx = list(range(args.training_label_num, args.training_label_num + args.training_unlabel_num))

    # 修改TwoStreamBatchSampler以支持DDP
    batch_sampler = DistributedTwoStreamBatchSampler(
        primary_indices=labeled_idx,
        secondary_indices=unlabeled_idx,
        batch_size=args.batch_size,
        secondary_batch_size=args.batch_size - args.label_bs, # 无标签批量大小
        rank=local_rank,
        num_replicas=world_size
    )

    # [DDP 修改] DataLoader不再需要sampler或shuffle，因为batch_sampler已经处理了
    train_loader = DataLoader(full_dataset, batch_sampler=batch_sampler, num_workers=os.cpu_count(), pin_memory=True)
    logging.info(f"[Rank {local_rank}] DataLoader created with DistributedTwoStreamBatchSampler.")

    # ==================== 模型初始化 ====================
    logging.info(f"[Rank {local_rank}] Initializing model...")
    model = Network(in_channels=1, num_classes=args.num_classes).to(device)
    
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model total parameters: {total_params:,}")
        logging.info(f"Model trainable parameters: {trainable_params:,}")

    start_epoch = 0
    if args.continue_train:
        if is_main_process: logging.info("Attempting to load latest checkpoint...")
        checkpoint_path = latest_checkpoint(args.pth_path)
        if checkpoint_path:
            checkpoint_data = torch.load(checkpoint_path, map_location=device)
            load_model(model, checkpoint_data['model_state_dict'], device)
            start_epoch = checkpoint_data['epoch'] + 1
            if is_main_process: logging.info(f"Loaded checkpoint: {checkpoint_path}. Resuming from epoch {start_epoch}")
        elif is_main_process:
            logging.warning("`continue_train` is set, but no checkpoint was found. Starting from scratch.")

    # [DDP 修改] 使用DDP封装模型
    # find_unused_parameters=True对于具有复杂分支的模型是安全的
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    logging.info(f"[Rank {local_rank}] Model wrapped with DDP.")

    # ==================== 优化器与损失函数 ====================
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    criterion_dice = CeDiceLoss(num_classes=args.num_classes, loss_weight=[0.6, 1.4])
    criterion_mse = softmax_mse_loss
    if is_main_process:
        logging.info(f"Optimizer: Adam, LR: {args.learning_rate}")
        logging.info("Loss functions: CeDiceLoss and softmax_mse_loss.")

    # ==================== TensorBoard (仅主进程) ====================
    writer = None
    if is_main_process:
        writer = SummaryWriter(log_dir=args.tensorboard_path)
        logging.info(f"TensorBoard writer initialized at: {args.tensorboard_path}")

    # ==================== 训练开始 ====================
    if is_main_process: logging.info(f"Starting training from epoch {start_epoch+1} to {args.epochs}...")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        # [DDP 修改] 必须在每个epoch开始时设置sampler的epoch，以确保shuffle
        batch_sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        epoch_dice_loss = 0.0
        epoch_consistency_loss = 0.0
        
        # [DDP 修改] 只有主进程显示tqdm进度条
        if is_main_process:
            pbar = tqdm.tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch+1}/{args.epochs}')
        else:
            pbar = train_loader

        for batch_idx, batch_data in enumerate(pbar):
            images = batch_data['image'].to(device, non_blocking=True)
            labels = batch_data['label'].to(device, non_blocking=True)
            labeled_bs = args.label_bs
            
            outputs = model(images)
            A_output, B_output = outputs[0], outputs[1]

            supervised_loss = criterion_dice(A_output[:labeled_bs], labels[:labeled_bs])
            consistency_loss = criterion_mse(A_output[labeled_bs:].detach(), B_output[labeled_bs:])
            total_loss = supervised_loss + args.consistency_weight * consistency_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # [DDP 修改] 使用 dist.all_reduce 同步所有GPU的损失值，以便进行准确的日志记录
            # 这可以确保所有GPU上的日志数值一致
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(supervised_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(consistency_loss, op=dist.ReduceOp.AVG)
            
            epoch_loss += total_loss.item()
            epoch_dice_loss += supervised_loss.item()
            epoch_consistency_loss += consistency_loss.item()
            
            # [DDP 修改] 只有主进程更新进度条和TensorBoard
            if is_main_process:
                pbar.set_postfix({
                    'Total Loss': f"{epoch_loss / (batch_idx + 1):.4f}",
                    'Dice Loss': f"{epoch_dice_loss / (batch_idx + 1):.4f}",
                    'Consistency Loss': f"{epoch_consistency_loss / (batch_idx + 1):.4f}"
                })
                
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/Total_Batch', total_loss.item(), global_step)
                writer.add_scalar('Loss/Dice_Batch', supervised_loss.item(), global_step)
                writer.add_scalar('Loss/Consistency_Batch', consistency_loss.item(), global_step)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

        # [DDP 修改] 只有主进程记录epoch日志和保存模型
        if is_main_process:
            avg_total_loss = epoch_loss / len(train_loader)
            avg_dice_loss = epoch_dice_loss / len(train_loader)
            avg_consistency_loss = epoch_consistency_loss / len(train_loader)
            
            logging.info(f"Epoch {epoch+1}/{args.epochs} finished. Avg Loss: {avg_total_loss:.4f}, Avg Dice: {avg_dice_loss:.4f}, Avg MSE: {avg_consistency_loss:.4f}")
            writer.add_scalar('Loss/Total_Epoch', avg_total_loss, epoch + 1)
            writer.add_scalar('Loss/Dice_Epoch', avg_dice_loss, epoch + 1)
            writer.add_scalar('Loss/Consistency_Epoch', avg_consistency_loss, epoch + 1)

            if (epoch + 1) % 20 == 0 or epoch == args.epochs - 1:
                state = {
                    'epoch': epoch,
                    # [DDP 修改] 保存 .module.state_dict()
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_total_loss,
                }
                save_path = os.path.join(args.pth_path, f"epoch_{epoch+1}.pth")
                torch.save(state, save_path)
                logging.info(f"Model checkpoint saved to: {save_path}")
    
    # ==================== 训练结束 ====================
    if is_main_process:
        writer.close()
        logging.info("Training complete!")
    
    # [DDP 修改] 清理分布式进程组
    dist.destroy_process_group()