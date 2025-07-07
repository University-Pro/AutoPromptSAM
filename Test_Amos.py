"""
Amos数据集的测试代码
但是现在这个版本只能测试一个输出结果
"""

import os
import argparse
import torch
import numpy as np
import SimpleITK as sitk
import logging
from datetime import datetime
from tqdm import tqdm
from medpy import metric
from typing import Tuple
from dataloader.DataLoader_Amos import AmosConfig, AmosDataset
from torch.cuda import get_device_name

# 导入对应的网络
# from networks.VNet_MultiOutput_V2 import VNet
from networks.VNet_MultiOutput_V3 import VNet

# 如果目录不存在则创建
def maybe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def setup_logging(log_file):
    """确保日志目录存在并配置日志记录"""
    log_dir = os.path.dirname(log_file)
    if log_dir:  # 防止空路径的情况
        os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除之前的handler防止重复
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# --- 核心推理函数（滑动窗口） ---
def test_single_case(net, image: np.ndarray, stride_xy: int, stride_z: int, patch_size: Tuple[int, int, int], num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    patch_d, patch_h, patch_w = patch_size
    img_d, img_h, img_w = image.shape
    
    # 添加批次和通道维度，转换为张量，发送到GPU
    input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
    _, _, d, h, w = input_tensor.shape
    
    score_map = torch.zeros((1, num_classes, d, h, w), dtype=torch.float32)
    count_map = torch.zeros((1, 1, d, h, w), dtype=torch.float32)
    
    steps_z = int(np.ceil((d - patch_d) / stride_z)) + 1
    steps_y = int(np.ceil((h - patch_h) / stride_xy)) + 1
    steps_x = int(np.ceil((w - patch_w) / stride_xy)) + 1
    
    with torch.no_grad():
        for iz in range(steps_z):
            sz = min(stride_z * iz, d - patch_d)
            for iy in range(steps_y):
                sy = min(stride_xy * iy, h - patch_h)
                for ix in range(steps_x):
                    sx = min(stride_xy * ix, w - patch_w)
                    patch = input_tensor[:, :, sz:sz + patch_d, sy:sy + patch_h, sx:sx + patch_w]
                    patch_pred = net(patch)

                    # 如果Patch的长度大于1，那么取第0个
                    if len(patch_pred) > 1:
                        patch_pred = patch_pred[0]

                    score_map[:, :, sz:sz + patch_d, sy:sy + patch_h, sx:sx + patch_w] += patch_pred.cpu()
                    count_map[:, :, sz:sz + patch_d, sy:sy + patch_h, sx:sx + patch_w] += 1.0
    
    count_map[count_map == 0] = 1.0
    score_map = score_map / count_map
    pred_map = torch.argmax(score_map, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    score_map_np = score_map.squeeze(0).cpu().numpy()  
    
    return pred_map, score_map_np

def calculate_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int) -> dict:
    metrics = {'Dice': 0.0, 'HD95': 0.0}
    for cls in range(1, num_classes):
        pred_cls = (pred == cls).astype(np.uint8)
        target_cls = (target == cls).astype(np.uint8)
        
        if np.sum(target_cls) == 0:
            continue
        
        dice = metric.binary.dc(pred_cls, target_cls)
        metrics['Dice'] += dice
        
        try:
            hd = metric.binary.hd95(pred_cls, target_cls)
            metrics['HD95'] += hd
        except RuntimeError:
            hd = np.nan
            
    valid_classes = num_classes - 1
    metrics['Dice'] /= valid_classes
    metrics['HD95'] /= valid_classes
    
    return metrics

if __name__ == '__main__':
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="在Amos数据集上测试VNet模型")
    parser.add_argument('--model_path', type=str, required=True,
                      help='预训练VNet .pth模型文件路径')
    parser.add_argument('--amos_data_path', type=str, default='./datasets/Amos',
                      help='Amos数据集根目录（包含test.txt和data文件夹）')
    parser.add_argument('--output_dir', type=str, default='./result/VNet/Amos/Test',
                      help='保存预测结果的目录（.nii.gz文件）')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'eval', 'train'],
                      help='用于测试的Amos数据集分割')
    parser.add_argument('--speed', type=int, default=0, choices=[0, 1, 2],
                      help='推理速度/步长设置（0: 慢/小步长，1: 中，2: 快/大步长）')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                      help='使用的GPU（例如 "0", "0,1"）')
    parser.add_argument('--n_filters', type=int, default=16, help='VNet基础滤波器数量')
    parser.add_argument('--num_channels', type=int, default=1, help='VNet输入通道数')
    parser.add_argument('--metrics_log', type=str, default='./result/metrics_log.txt',
                      help='评估指标日志文件路径')
    parser.add_argument('--save_images', action='store_true',
                      help='是否保存预测结果图像')
    args = parser.parse_args()

    # ==================== 初始化设置 ====================
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 确保目录存在
    maybe_mkdir(args.output_dir)
    maybe_mkdir(os.path.dirname(args.metrics_log))

    # ==================== 日志系统初始化 ====================
    setup_logging(args.metrics_log)
    logging.info("\n" + "="*40 + " 运行配置 " + "="*40)
    logging.info(f"PyTorch 版本: {torch.__version__}")
    logging.info(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA 设备数量: {torch.cuda.device_count()}")
        logging.info(f"当前设备: {torch.cuda.current_device()} - {get_device_name(0)}")
        logging.info(f"设备内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    logging.info(f"工作设备: {device}")
    
    # 记录完整参数
    logging.info("\n" + "-"*20 + " 实验参数 " + "-"*20)
    for arg in vars(args):
        logging.info(f"{arg:20}: {getattr(args, arg)}")
    logging.info(f"{'启动时间':20}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ==================== 数据集配置 ====================
    try:
        amos_config = AmosConfig(
            save_dir=args.amos_data_path,
            # patch_size=(80, 160, 160)  # 全尺寸设置
            patch_size=(160,160,80)  # 全尺寸设置
        )
        patch_size = amos_config.patch_size
        num_classes = amos_config.num_classes
        
        # 步长配置
        stride_config = {
            0: (patch_size[1]//4, patch_size[0]//4),
            1: (patch_size[1]//2, patch_size[0]//2),
            2: (patch_size[1], patch_size[0])
        }
        stride_xy, stride_z = [max(1, x) for x in stride_config[args.speed]]
        
        logging.info("\n" + "-"*20 + " 数据配置 " + "-"*20)
        logging.info(f"输入尺寸: {patch_size}")
        logging.info(f"类别数量: {num_classes}")
        logging.info(f"滑动窗口步长: XY={stride_xy}, Z={stride_z}")
    except Exception as e:
        logging.exception("数据集配置失败!")
        raise

    # ==================== 模型初始化 ====================
    try:
        model = VNet(
            n_channels=args.num_channels,
            n_classes=num_classes,
            n_filters=args.n_filters,
            normalization='batchnorm',
            has_dropout=True
        ).to(device)
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logging.info("\n" + "-"*20 + " 模型架构 " + "-"*20)
        logging.info(f"总参数量: {total_params/1e6:.2f}M")
        logging.info(f"可训练参数: {trainable_params/1e6:.2f}M")

        # 加载权重
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=device)
            
            # 检查模型参数键是否存在
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"成功加载模型参数 from {args.model_path}")
                
                # 记录训练元数据（如果有的话）
                if 'epoch' in checkpoint:
                    logging.info(f"训练轮次: {checkpoint['epoch']}")
                if 'best_dice' in checkpoint:
                    logging.info(f"最佳验证Dice: {checkpoint['best_dice']:.4f}")
            else:
                # 兼容旧版模型，假设整个checkpoint是model_state_dict
                model.load_state_dict(checkpoint)
                logging.warning("加载未包含元数据的旧版模型参数")
        else:
            raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
            
        model.eval()
        logging.info("模型已设置为评估模式")
    except Exception as e:
        logging.exception("模型初始化失败!")
        raise

    # ==================== 数据加载 ====================
    try:
        temp_dataset = AmosDataset(split=args.split, config=amos_config)
        test_ids = temp_dataset.ids_list
        logging.info("\n" + "="*40 + " 数据统计 " + "="*40)
        logging.info(f"加载数据集分割: {args.split}")
        logging.info(f"案例总数: {len(test_ids)}")
        logging.info(f"首案例ID示例: {test_ids[0] if test_ids else '无'}")
    except Exception as e:
        logging.exception("数据加载失败!")
        raise

    # ==================== 推理流水线 ====================
    total_metrics = {'Dice': [], 'HD95': []}
    logging.info("\n" + "="*40 + " 开始推理 " + "="*40)
    
    try:
        with torch.no_grad():
            progress_bar = tqdm(test_ids, desc=f"处理{args.split}数据集")
            for case_id in progress_bar:
                try:
                    # 加载数据
                    image, label = temp_dataset._load_sample(case_id)
                    
                    # 推理
                    pred, _ = test_single_case(
                        model, image, 
                        stride_xy, stride_z,
                        patch_size, num_classes
                    )
                    
                    # 指标计算
                    dice = metric.binary.dc(pred > 0, label > 0)
                    try:
                        hd = metric.binary.hd95(pred > 0, label > 0)
                    except RuntimeError:
                        hd = np.nan
                        logging.warning(f"{case_id} HD95计算失败")
                    
                    # 记录结果
                    total_metrics['Dice'].append(dice)
                    total_metrics['HD95'].append(hd)
                    
                    # 日志记录
                    logging.info(
                        f"案例 {case_id} | "
                        f"Dice: {dice:.4f} | "
                        f"HD95: {hd:.2f}" if not np.isnan(hd) else "HD95: NaN" + " | "
                        f"预测形状: {pred.shape} | "
                        f"类别分布: {np.unique(pred, return_counts=True)[1].tolist()}"
                    )
                    
                    # 保存结果
                    if args.save_images:
                        sitk_img = sitk.GetImageFromArray(pred.astype(np.uint8))
                        sitk.WriteImage(sitk_img, os.path.join(args.output_dir, f"{case_id}_pred.nii.gz"))
                        
                except Exception as e:
                    logging.error(f"案例 {case_id} 处理失败: {str(e)}", exc_info=True)
                    continue
    except KeyboardInterrupt:
        logging.warning("用户中断执行!")
        raise
    finally:
        # ==================== 最终统计 ====================
        logging.info("\n" + "="*40 + " 最终统计 " + "="*40)
        valid_count = len(total_metrics['Dice'])
        logging.info(f"成功处理案例数: {valid_count}/{len(test_ids)} ({valid_count/len(test_ids):.1%})")
        
        # Dice统计
        dice_array = np.array(total_metrics['Dice'])
        logging.info(f"[Dice] 均值 ± 标准差: {np.nanmean(dice_array):.4f} ± {np.nanstd(dice_array):.4f}")
        logging.info(f"[Dice] 中位数 (IQR): {np.nanmedian(dice_array):.4f} "
                    f"({np.nanquantile(dice_array, 0.25):.4f} - {np.nanquantile(dice_array, 0.75):.4f})")
        
        # HD95统计
        hd_array = np.array(total_metrics['HD95'])
        logging.info(f"[HD95] 均值 ± 标准差: {np.nanmean(hd_array):.2f} ± {np.nanstd(hd_array):.2f}")
        logging.info(f"[HD95] 中位数 (IQR): {np.nanmedian(hd_array):.2f} "
                    f"({np.nanquantile(hd_array, 0.25):.2f} - {np.nanquantile(hd_array, 0.75):.2f})")
        
        # 记录显存使用
        if torch.cuda.is_available():
            logging.info(f"峰值显存使用: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
            logging.info(f"当前显存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        logging.info("测试流程完成!")