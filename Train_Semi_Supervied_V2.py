"""
尝试双流编码器是否真的有作用
用一个伪标签的EMA的进行尝试
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

# LA的DataLoader
# from dataloader.DataLoader_LA import LAHeart
from dataloader.DataLoader_LA_Semi import LAHeart


# 导入LA的数据增强
from utils.ImageAugment import RandomRotFlip_LA as RandomRotFlip
from utils.ImageAugment import RandomCrop_LA as RandomCrop
from utils.ImageAugment import ToTensor_LA as ToTensor

# 导入加载无标签的工具
# 这里不适用双流编码器，而是手动创建batch
# from utils.ImageAugment import TwoStreamBatchSampler_LA
# from utils.ImageAugment import TwoStreamBatchSampler_LA_v2 

# 导入网络框架
from networks.VNet import VNet

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

def update_ema_model(ema_model, model, alpha=0.999):
    # 指数移动平均更新
    for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(model_param.data, alpha=1-alpha)

# 在训练循环中添加函数定义
def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(iter):
    return 0.1 * sigmoid_rampup(iter, 200)  # 假设总迭代次数为200

if __name__ == "__main__":
    # ==================== 参数配置 ====================
    parser = argparse.ArgumentParser(description="3D医学图像分割训练脚本")

    # 训练参数
    parser.add_argument('--seed', type=int, default=21, help='随机种子')
    parser.add_argument("--epochs", type=int, default=200, help='训练总轮数')
    parser.add_argument("--batch_size", type=int, default=4, help='批量大小')
    parser.add_argument('--num_classes', type=int, default=2, help='分类数量')
    parser.add_argument("--learning_rate", type=float,
                        default=1e-5, help='初始学习率')

    # 路径参数
    parser.add_argument('--dataset_path', type=str,
                        default='./datasets/LA', help='数据集路径')
    parser.add_argument("--log_path", type=str,
                        default="./result/Training.log", help='日志文件路径')
    parser.add_argument("--pth_path", type=str,
                        default='./result/UNet2D_VNet3D/LA/Pth', help='模型保存路径')
    parser.add_argument("--tensorboard_path", type=str,
                        default='./result/Train', help='TensorBoard日志路径')

    # 模型参数
    parser.add_argument("--continue_train", action="store_true", help='继续训练')
    parser.add_argument("--multi_gpu", action="store_true", help='使用多GPU训练')
    parser.add_argument("--training_label_num", type=int, default=16, help='有标签样本数量')
    parser.add_argument("--training_unlabel_num", type=int, default=64, help='无标签样本数量')

    args = parser.parse_args()

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

    # 有标签数据集
    train_label_dataset = LAHeart(
        base_dir=args.dataset_path,
        split='train_label',
        num=args.training_label_num,  # 修正参数名
        transform=train_transform
    )

    train_unlabel_dataset = LAHeart(
        base_dir=args.dataset_path,
        split='train_unlabel',
        num=args.training_unlabel_num,  # 修正参数名
        transform=train_transform
    )
    
    # 创建对应的Dataloader（添加batch_size参数）
    train_label_loader = DataLoader(
        train_label_dataset,
        batch_size=args.batch_size // 2,  # 假设每个batch一半是标注数据
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    train_unlabel_loader = DataLoader(
        train_unlabel_dataset,
        batch_size=args.batch_size - (args.batch_size // 2),  # 剩余为无标注数据
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # ==================== 模型配置 ====================
    # 初始化模型
    model = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True)
    ema_model = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True)

    # 设置ema模型参数无法更新，只能通过ema的方式更新
    for param in ema_model.parameters():
            param.detach_()

    # 多GPU配置（如果启用多GPU并且设备数量大于1，则包装为DataParallel）
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        ema_model = nn.DataParallel(ema_model)
        logging.info(f"使用 {torch.cuda.device_count()} 个GPU进行训练")

    # 继续训练配置（从最近的检查点加载模型权重）
    if args.continue_train:
        checkpoint = latest_checkpoint(args.pth_path)
        if checkpoint:
            load_model(model, checkpoint, device)
            logging.info(f"从检查点恢复训练: {checkpoint}")

    model.to(device)
    ema_model.to(device)

    # ==================== 训练配置 ====================
    # 优化器配置（使用SGD优化器，并设置学习率、动量、权重衰减等参数）
    optimizer = optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    
    # 损失函数设置：
    criterion_dice = CeDiceLoss(num_classes=args.num_classes)
    criterion_mse = softmax_mse_loss
    
    # TensorBoard配置（记录训练过程中的各项指标）
    writer = SummaryWriter(log_dir=args.tensorboard_path)

    # ==================== 训练循环 ====================
    logging.info("开始训练...")
    # 初始化iter_num
    iter_num = 0
    max_iters = args.epochs * max(len(train_label_loader), len(train_unlabel_loader))  # 动态计算最大
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        ema_model.eval()  # EMA模型始终在eval模式

        epoch_loss = 0.0
        epoch_dice_loss = 0.0
        epoch_mse_loss = 0.0

        # 使用zip对齐两个loader的迭代
        pbar = tqdm.tqdm(zip(train_label_loader, train_unlabel_loader),
                        total=min(len(train_label_loader), len(train_unlabel_loader)),
                        desc=f'Epoch {epoch+1}/{args.epochs}',
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        for batch_idx, (labeled_batch, unlabeled_batch) in enumerate(pbar):

            # 合并标注和非标注数据
            images = torch.cat([labeled_batch['image'], unlabeled_batch['image']], dim=0).to(device)
            labels = torch.cat([labeled_batch['label'], torch.zeros_like(unlabeled_batch['label'])], dim=0).to(device)

            # 检查batch组成
            print(f"Batch {batch_idx}:")
            print(f"  Labeled samples: {images[:args.labeled_bs].shape}")    # 应为 [2, 1, 112, 112, 80]
            print(f"  Unlabeled samples: {images[args.labeled_bs:].shape}")  # 应为 [2, 1, 112, 112, 80]

            # 修改这里：正确获取无标签数据
            unlabeled_images = images[args.labeled_bs:] 
            # unlabeled_labels = labels[args.labeled_bs:]  # 不应该使用unlabeled_labeles，因为它们是无标签的

            # 无标签数据进行数据增强
            noise = torch.clamp(torch.randn_like(unlabeled_images) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_images + noise

            # 有标签数据进行向前传播
            outputs = model(images)

            # 无标签数据向前传播
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)

            # Mean Teacher的核心创新点
            Time = 8 # 增强次数
            volume_batch_r = unlabeled_images.repeat(2,1,1,1,1) # 重复数据
            stride = volume_batch_r.shape[0] // 2 # 步长
            preds = torch.zeros([stride * Time, 2, 112, 112, 80]).to(device=device)  # 存储所有增强预测
            
            # 多轮扰动预测
            for i in range(Time//2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
            
            preds = F.softmax(preds, dim=1)  # 对每轮预测计算softmax [T*B, C, D, H, W]
            preds = preds.reshape(Time, stride, 2, 112, 112, 80)  # 重组为 [T, B, C, D, H, W]
            preds = torch.mean(preds, dim=0)  # 沿T维度求平均 → [B, C, D, H, W]
            # 不确定性估计
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)  # [B, 1, D, H, W]

            # 计算监督学习损失函数
            supervised_loss = criterion_dice(outputs[:args.labeled_bs],labels[:args.labeled_bs]) # 防止计算无标签的错误梯度

            # 计算一致性距离（教师模型和学生模型预测之间的差异）
            # 首先获取学生模型对无标签数据的预测
            with torch.no_grad():
                student_outputs = model(unlabeled_images)
            student_outputs_soft = F.softmax(student_outputs, dim=1)
            
            # 计算一致性距离（MSE损失）
            consistency_dist = criterion_mse(student_outputs_soft, ema_output)  # 使用之前定义的softmax_mse_loss
            
            # 计算半监督学习损失函数
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            threshold = (0.75 + 0.25 * sigmoid_rampup(iter_num, 200)) * np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_loss = consistency_weight * torch.sum(mask * consistency_dist) / (torch.sum(mask) + 1e-16)
            iter_num += 1  # 每个batch更新计数器
            
            # 合并损失函数
            total_loss = supervised_loss + consistency_loss

            # 反向传播并跟新EMA
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 更新EMA模型
            # 确保 update_ema_model 函数能正确处理 DataParallel 包装的模型
            update_ema_model(ema_model, model.module if hasattr(model, 'module') else model)

            # 记录损失
            epoch_loss += total_loss.item()
            # 注意：如果dice_loss只在部分批次或样本上计算，直接累加可能不准确地反映平均值
            epoch_dice_loss += supervised_loss.item() if labels is not None else 0.0
            epoch_mse_loss += consistency_loss.item()

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                'Loss': f"{epoch_loss/(batch_idx+1):.4f}",
                'Dice': f"{epoch_dice_loss/(batch_idx+1):.4f}", # 这个平均值可能受标签存在与否影响
                'MSE': f"{epoch_mse_loss/(batch_idx+1):.4f}"
            })

            # TensorBoard记录
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Total', total_loss.item(), global_step)
            if labels is not None: # 只在有标签时记录Dice Loss才有意义
                writer.add_scalar('Loss/Dice', supervised_loss.item(), global_step)
            writer.add_scalar('Loss/MSE', consistency_loss.item(), global_step)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], global_step) # 记录学习率

        # Epoch 结束
        pbar.close()
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_dice_loss = epoch_dice_loss / len(train_loader) # 或更精确地除以有标签的批次数
        avg_mse_loss = epoch_mse_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} completed. Avg Loss: {avg_epoch_loss:.4f}, Avg Dice: {avg_dice_loss:.4f}, Avg MSE: {avg_mse_loss:.4f}")

        # 保存检查点 (保持不变)
        if (epoch + 1) % 20 == 0 or epoch == args.epochs - 1:
            state = {
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            save_path = os.path.join(args.pth_path, f"epoch_{epoch+1}.pth")
            torch.save(state, save_path)
            logging.info(f"模型已保存至: {save_path}")

    writer.close()
    logging.info("训练完成！")