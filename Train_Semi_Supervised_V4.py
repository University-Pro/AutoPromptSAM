"""
LA数据集
使用双流编码器
进行3D监督/半监督伪标签训练
可以通过设置有标签和无标签数据的多少来确定是否为半监督或者是监督
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
from dataloader.DataLoader_LA import LAHeart

# 导入LA的数据增强
from utils.ImageAugment import RandomRotFlip_LA as RandomRotFlip
from utils.ImageAugment import RandomCrop_LA as RandomCrop
from utils.ImageAugment import ToTensor_LA as ToTensor

# 导入加载无标签的工具
from utils.ImageAugment import TwoStreamBatchSampler_LA

# 导入网络框架
# from networks.VNet import VNet
# from networks.SAM3D_VNet_SSL import Network
# from networks.SAM3D_VNet_SSL_V3 import Network
from networks.SAM3D_VNet_SSL_V1 import Network

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

# 查找最新的文件
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

if __name__ == "__main__":
    # ==================== 监督学习参数配置 ====================
    parser = argparse.ArgumentParser(description="3D医学图像分割训练脚本（半监督）")
    parser.add_argument('--seed', type=int, default=21, help='随机种子')
    parser.add_argument("--epochs", type=int, default=200, help='训练总轮数')
    parser.add_argument("--batch_size", type=int, default=4, help='总批量大小（有+无标签）')
    parser.add_argument('--num_classes', type=int, default=2, help='类别数量')
    parser.add_argument("--learning_rate", type=float, default=1e-5, help='初始学习率')
    parser.add_argument('--dataset_path', type=str, default='./datasets/LA', help='数据集路径')
    parser.add_argument("--log_path", type=str, default="./result/Training.log", help='日志文件路径')
    parser.add_argument("--pth_path", type=str, default='./result/VNet/LA/Pth', help='模型保存路径')
    parser.add_argument("--tensorboard_path", type=str, default='./result/Train', help='TensorBoard日志路径')
    parser.add_argument("--continue_train", action="store_true", help='是否继续训练')
    parser.add_argument("--multi_gpu", action="store_true", help='是否使用多GPU')
    
    # ==================== 半监督学习参数配置 ====================
    parser.add_argument("--training_label_num", type=int, default=16, help='有标签样本数量')
    parser.add_argument("--training_unlabel_num", type=int, default=64, help='无标签样本数量')
    parser.add_argument("--label_bs", type=int, default=2, help='有标签样本批量大小')
    parser.add_argument("--unlabel_bs", type=int, default=2, help='无标签样本批量大小')
    parser.add_argument("--consistency_weight", type=float, default=0.1, help='一致性损失权重') 
    
    # ==================== 预训练权重加载 ====================
    parser.add_argument("--pretrained_weights", type=str, default=None, help='预训练权重路径')
    args = parser.parse_args()
    
    # ==================== 文件与目录创建 ====================
    for path in [os.path.dirname(args.log_path), args.pth_path, args.tensorboard_path]:
        os.makedirs(path, exist_ok=True)

    # ==================== 初始化设置 ====================
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_logging(args.log_path)
    logging.info(f"训练配置参数:\n{vars(args)}")

    # ==================== 数据准备 ====================
    patch_size = (112, 112, 80)
    train_transform = transforms.Compose([
        RandomRotFlip(),
        RandomCrop(patch_size),
        ToTensor()
    ])
    full_dataset = LAHeart(base_dir=args.dataset_path, split='train', transform=train_transform)
    
    labeled_idx = list(range(args.training_label_num))
    unlabeled_idx = list(range(args.training_label_num, args.training_label_num + args.training_unlabel_num))

    batch_sampler = TwoStreamBatchSampler_LA(
        primary_indices=labeled_idx,
        secondary_indices=unlabeled_idx,
        batch_size=args.batch_size,
        secondary_batch_size=args.unlabel_bs
    )

    train_loader = DataLoader(full_dataset, batch_sampler=batch_sampler, num_workers=8, pin_memory=True)

    # ==================== 模型初始化 ====================
    # model = UNet_Full(n_channels=1, n_classes=args.num_classes).to(device)
    model = Network(pretrain_weight_path="./result/VNet/LA/Pth/best.pth").to(device=device)
    # model = Network(pretrain_weight_path="./result/VNet/LA/Pth/best.pth",encoder_depth=8).to(device=device)
    
    # 冻结Network的ImageEncoder3D模块
    for param in model.samencoder.parameters():
        param.requires_grad = False

    # 多GPU支持
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logging.info(f"使用 {torch.cuda.device_count()} 个GPU")

    # 记录模型的参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"模型总参数: {total_params:,}")
    logging.info(f"可训练参数: {trainable_params:,}")

    # 继续训练加载模型权重
    if args.continue_train:
        checkpoint = latest_checkpoint(args.pth_path)
        if checkpoint:
            # 加载整个检查点数据
            checkpoint_data = torch.load(checkpoint, map_location=device)
            # 模型的状态字典并加载
            model_state_dict = checkpoint_data['model_state_dict']
            load_model(model, model_state_dict, device)
            # 恢复epoch
            start_epoch = checkpoint_data['epoch'] + 1
            logging.info(f"加载检查点: {checkpoint}，从epoch {start_epoch}继续训练")
    
    # 加载预训练权重
    # model = load_pretrained_to_branch1(model, args.pretrained_weights,multi_gpu=args.multi_gpu)

    # 将模型移动到GPU
    model.to(device)

    # 优化器设置
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5) # 使用Adam优化器

    # 损失函数
    criterion_dice = CeDiceLoss(num_classes=args.num_classes)
    criterion_mse = softmax_mse_loss
    celoss = nn.CrossEntropyLoss()

    # TensorBoard
    writer = SummaryWriter(log_dir=args.tensorboard_path)

    # ==================== 训练开始 ====================
    iter_num = 0
    max_iters = args.epochs * len(train_loader)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice_loss = 0.0
        epoch_consistency_loss = 0.0
        
        pbar = tqdm.tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_idx, batch_data in enumerate(pbar):
            # 数据准备
            images = batch_data['image'].to(device)
            labels = batch_data['label'].to(device)
            labeled_bs = args.label_bs
            
            # 模型前向传播（双输出）
            outputs = model(images)
            A_output = outputs[0]  # 主分支输出
            B_output = outputs[1]  # 辅助分支输出

            # 监督损失（仅计算有标签数据）
            supervised_loss = criterion_dice(A_output[:labeled_bs], labels[:labeled_bs])
            
            # 一致性损失（无标签数据，A指导B）
            consistency_loss = criterion_mse(
                A_output[labeled_bs:].detach(),  # 使用detach防止梯度传播到A
                B_output[labeled_bs:]
            )
            
            # 总损失 = 监督损失 + 一致性权重 * 一致性损失
            total_loss = supervised_loss + args.consistency_weight * consistency_loss

            # 反向传播与优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 统计损失
            epoch_loss += total_loss.item()
            epoch_dice_loss += supervised_loss.item()
            epoch_consistency_loss += consistency_loss.item()
            
            # 更新进度条显示
            pbar.set_postfix({
                'Total Loss': f"{epoch_loss / (batch_idx + 1):.4f}",
                'Dice Loss': f"{epoch_dice_loss / (batch_idx + 1):.4f}",
                'Consistency Loss': f"{epoch_consistency_loss / (batch_idx + 1):.4f}"
            })
            
            # TensorBoard日志记录
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Total', total_loss.item(), global_step)
            writer.add_scalar('Loss/Dice', supervised_loss.item(), global_step)
            writer.add_scalar('Loss/Consistency', consistency_loss.item(), global_step)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], global_step)
            iter_num += 1

        # 记录epoch
        logging.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/(batch_idx+1):.4f}, Dice: {epoch_dice_loss/(batch_idx+1):.4f}, MSE: {epoch_consistency_loss/(batch_idx+1):.4f}")

        # 保存模型
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(train_loader),
            }
            torch.save(state, os.path.join(args.pth_path, f"epoch_{epoch+1}.pth"))
            logging.info(f"模型已保存至: epoch_{epoch+1}.pth")
    
    writer.close()
    logging.info("训练完成！")