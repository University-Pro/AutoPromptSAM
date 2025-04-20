"""
半监督训练相关代码
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
from torch.optim.lr_scheduler import StepLR # 动态学习率
from torch.utils.tensorboard import SummaryWriter # 启用Tensorboard
import logging # 日志系统
import argparse
from glob import glob

# LA的DataLoader
from dataloader.DataLoader_LA import LAHeart

# 导入LA的数据增强
from utils.ImageAugment import RandomRotFlip_LA as RandomRotFlip
from utils.ImageAugment import RandomCrop_LA as RandomCrop
from utils.ImageAugment import ToTensor_LA as ToTensor

# 导入加载无标签的工具
from utils.ImageAugment import TwoStreamBatchSampler_LA

# 导入网络框架
from networks.Vit3DUNet3D import UNet3D

# 导入Loss函数
from utils.LA_Train_Metrics import softmax_mse_loss
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

# 查找最新的文件
def latest_checkpoint(path):
    """在path中查找出最新的文件"""
    list_of_files = glob(os.path.join(path, '*.pth'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# 自动处理多卡和单卡模型
def load_model_deprecated(model, model_path, device):
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

def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # 判断当前模型是否是 DataParallel 实例
    is_dataparallel = isinstance(model, nn.DataParallel)
    
    # 动态处理键名前缀
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
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

if __name__ == "__main__":
    # ==================== 参数配置 ====================
    parser = argparse.ArgumentParser(description="3D医学图像分割训练脚本")
    
    # 训练参数
    parser.add_argument('--seed', type=int, default=21, help='随机种子')
    parser.add_argument("--epochs", type=int, default=200, help='训练总轮数')
    parser.add_argument("--batch_size", type=int, default=4, help='批量大小')
    parser.add_argument('--num_classes', type=int, default=2, help='分类数量')
    parser.add_argument("--learning_rate", type=float, default=1e-5, help='初始学习率')
    
    # 路径参数
    parser.add_argument('--dataset_path', type=str, default='./datasets/LA', help='数据集路径')
    parser.add_argument("--log_path", type=str, default="./result/Training.log", help='日志文件路径')
    parser.add_argument("--pth_path", type=str, default='./result/UNet2D_VNet3D/LA/Pth', help='模型保存路径')
    parser.add_argument("--tensorboard_path", type=str, default='./result/Train', help='TensorBoard日志路径')
    
    # 模型参数
    parser.add_argument("--continue_train", action="store_true", help='继续训练')
    parser.add_argument("--multi_gpu", action="store_true", help='使用多GPU训练')
    parser.add_argument("--training_num", type=int, default=80, help='有标签样本数量')
    parser.add_argument("--labeled_bs", type=int, default=2, help='每批有标签样本数')
    
    # SAM相关参数（如果不使用SAM模型，这里可以忽略）
    parser.add_argument("--sam2_checkpoint", type=str, 
                       default='./sam2_configs/sam2.1_hiera_tiny.pt', 
                       help='SAM2预训练权重路径')
    parser.add_argument("--model_cfg", type=str, 
                       default='sam2.1/sam2.1_hiera_t.yaml',
                       help='SAM2模型配置文件路径')
    
    # 混合精度训练参数
    parser.add_argument("--mixed_precision", action="store_true", help='使用混合精度训练')
    
    args = parser.parse_args()

    # ==================== 目录检查与创建 ====================
    # 获取日志文件目录，并创建（如果不存在）
    log_dir = os.path.dirname(args.log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        print(f"创建日志文件目录: {log_dir}")

    # 创建模型保存目录
    if not os.path.exists(args.pth_path):
        os.makedirs(args.pth_path, exist_ok=True)
        print(f"创建模型保存目录: {args.pth_path}")

    # 创建TensorBoard日志目录
    if not os.path.exists(args.tensorboard_path):
        os.makedirs(args.tensorboard_path, exist_ok=True)
        print(f"创建TensorBoard日志目录: {args.tensorboard_path}")

    # ==================== 初始化设置 ====================
    # 设置随机种子（保证结果可重复）
    torch.manual_seed(args.seed)
    
    # 设备配置（优先使用GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 日志系统初始化（假设setup_logging为自定义日志设置函数）
    setup_logging(args.log_path)
    logging.info(f"训练配置参数:\n{vars(args)}")

    # ==================== 数据准备 ====================
    # 数据增强配置（根据实际需求进行数据预处理）
    patch_size = (112, 112, 80)
    train_transform = transforms.Compose([
        RandomRotFlip(),   # 随机旋转翻转
        RandomCrop(patch_size),  # 随机裁剪
        ToTensor(),        # 转换为Tensor格式
    ])
    
    # 数据集加载（LAHeart为数据集类，根据实际情况调整）
    train_dataset = LAHeart(
        base_dir=args.dataset_path,
        split='train',
        transform=train_transform
    )
    
    # 双流采样器（半监督学习场景下，有标签与无标签样本分开采样）
    labeled_idxs = list(range(args.training_num))
    unlabeled_idxs = list(range(args.training_num, 80))
    batch_sampler = TwoStreamBatchSampler_LA(
        labeled_idxs, 
        unlabeled_idxs, 
        args.batch_size, 
        args.batch_size - args.labeled_bs
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=8,
        pin_memory=True
    )

    # ==================== 模型配置 ====================
    # 初始化模型（这里使用UNet2D_VNet3D作为示例模型）
    model = UNet3D(in_channels=1,out_channels=2).to(device=device)

    # 多GPU配置（如果启用多GPU并且设备数量大于1，则包装为DataParallel）
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logging.info(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
    
    # 继续训练配置（从最近的检查点加载模型权重）
    if args.continue_train:
        checkpoint = latest_checkpoint(args.pth_path)
        if checkpoint:
            load_model(model, checkpoint, device)
            logging.info(f"从检查点恢复训练: {checkpoint}")
    
    model.to(device)
    
    # ==================== 训练配置 ====================
    # 优化器配置（使用SGD优化器，并设置学习率、动量、权重衰减等参数）
    optimizer = optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    
    # 损失函数设置：
    # 1. Dice损失用于监督学习部分（真实标签与vnet_output）
    # 2. softmax_mse_loss用于半监督学习中，衡量两个output之间的差异
    criterion_dice = CeDiceLoss(num_classes=args.num_classes)
    criterion_mse = softmax_mse_loss
    
    # 学习率调度器（使用余弦退火调度器）
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,
        T_mult=2,
        eta_min=1e-6
    )
    
    # TensorBoard配置（记录训练过程中的各项指标）
    writer = SummaryWriter(log_dir=args.tensorboard_path)
    
    # 混合精度训练配置
    if args.mixed_precision:
        scaler = torch.amp.GradScaler()
        logging.info("使用混合精度训练")
    else:
        scaler = None
        logging.info("不使用混合精度训练")

    # ==================== 训练循环 ====================
    logging.info("开始训练...")
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_dice_loss = 0.0
        epoch_mse_loss = 0.0
        
        # 配置进度条显示当前训练进度
        pbar = tqdm.tqdm(total=len(train_loader), 
                         desc=f'Epoch {epoch+1}/{args.epochs}',
                         bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        
        for batch_idx, batch_data in enumerate(train_loader):
            # 获取图像和标签数据，并转移至设备（GPU或CPU）
            images = batch_data['image'].to(device)
            labels = batch_data['label'].to(device)
            
            # 梯度清零，防止梯度累加
            optimizer.zero_grad()
            
            if args.mixed_precision:
                # 混合精度前向传播
                with torch.amp.autocast(device_type=device):
                    # 网络输出：vnet_output用于监督损失，unet_output作为伪标签
                    vnet_output, unet_output = model(images)
                    
                    # 计算Dice损失（真实标签与vnet_output之间）
                    dice_loss = criterion_dice(vnet_output, labels)
                    
                    # 伪标签损失计算（使用unet_output作为伪标签进行softmax后计算MSE损失）
                    probs = torch.softmax(unet_output, dim=1)  # 计算伪标签概率分布
                    
                    # 使用置信度过滤，只选择概率最高值大于0.8的部分参与损失计算
                    confidence_mask = (torch.max(probs, dim=1, keepdim=True)[0] > 0.7)
                    
                    if confidence_mask.sum() > 0:
                        mse_loss = criterion_mse(unet_output * confidence_mask, vnet_output)
                    else:
                        mse_loss = torch.tensor(0.0, device=device)
                    
                    # 动态调整损失权重（随着训练逐渐增加伪标签损失的影响）
                    alpha = min(0.5, 0.1 + (epoch / args.epochs) * 0.4)
                    total_loss = dice_loss + alpha * mse_loss
            else:
                # 损失函数设计：结合监督学习和半监督学习部分
                criterion_dice = CeDiceLoss(num_classes=args.num_classes)
                criterion_mse = softmax_mse_loss

                # 动态权重，控制伪标签的影响（可以通过alpha权重平滑过渡）
                alpha = min(0.5, 0.1 + (epoch / args.epochs) * 0.4)

                # 在训练初期，更多依赖真实标签的Dice Loss，逐渐引入伪标签
                dice_weight = 1.0 - alpha
                mse_weight = alpha

                # ==================== 损失计算 ====================
                unet_output,vnet_output = model(images)

                # 监督学习部分（Dice Loss）
                dice_loss = criterion_dice(vnet_output, labels)

                # 半监督学习部分（伪标签MSE Loss）
                # probs = torch.softmax(unet_output, dim=1)  # 计算伪标签概率分布
                # confidence_mask = (torch.max(probs, dim=1, keepdim=True)[0] > 0.7)  # 增加置信度阈值，避免低质量伪标签

                # if confidence_mask.sum() > 0:
                #     mse_loss = criterion_mse(unet_output * confidence_mask, vnet_output)
                # else:
                #     mse_loss = torch.tensor(0.0, device=device)


                # 不使用置信度过滤，直接计算MSE Loss
                mse_loss = criterion_mse(unet_output, vnet_output)

                # 总损失计算：加权的Dice Loss和MSE Loss
                total_loss = dice_weight * dice_loss + mse_weight * mse_loss

            # 反向传播与参数更新
            if args.mixed_precision:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                # 裁剪梯度，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # 累加损失值，用于监控训练过程
            epoch_loss += total_loss.item()
            epoch_dice_loss += dice_loss.item()
            epoch_mse_loss += mse_loss.item()
            
            # 记录训练过程中的损失值到TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Total', total_loss.item(), global_step)
            writer.add_scalar('Loss/Dice', dice_loss.item(), global_step)
            writer.add_scalar('Loss/MSE', mse_loss.item(), global_step)
            
            # 更新进度条显示信息
            pbar.set_postfix({
                'Loss': f"{epoch_loss/(batch_idx+1):.4f}",
                'Dice': f"{epoch_dice_loss/(batch_idx+1):.4f}",
                'MSE': f"{epoch_mse_loss/(batch_idx+1):.4f}"
            })
            pbar.update(1)
        
        pbar.close()
        # 调整学习率
        scheduler.step()
        
        # 日志记录当前Loss
        logging.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/(batch_idx+1):.4f}, Dice: {epoch_dice_loss/(batch_idx+1):.4f}, MSE: {epoch_mse_loss/(batch_idx+1):.4f}")

        # 每隔固定epoch保存一次模型检查点（确保目录存在）
        # 虽然前面已经创建了目录，这里仍可做二次检查，保证万无一失
        if (epoch + 1) % 20 == 0 or epoch == args.epochs - 1:
            # 确保保存目录存在
            if not os.path.exists(args.pth_path):
                os.makedirs(args.pth_path, exist_ok=True)
            save_path = os.path.join(args.pth_path, f"epoch_{epoch+1}.pth")
            try:
                torch.save(model.state_dict(), save_path)
                logging.info(f"模型已保存至: {save_path}")
            except Exception as e:
                logging.error(f"保存模型时出错: {e}")
            
    # 训练结束后关闭TensorBoard记录器
    writer.close()
    logging.info("训练完成！")