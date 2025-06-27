"""
这个版本可以处理拥有多个输出结果的网络
并且计算平均值
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
from typing import Tuple, Dict, List
from dataloader.DataLoader_Amos import AmosConfig, AmosDataset
from torch.cuda import get_device_name
from collections import OrderedDict

# 导入相关网络
from networks.Double_VNet import Network

# 如果目录不存在则创建
def maybe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def setup_logging(log_file):
    """确保日志目录存在并配置日志记录"""
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_file, mode='a')
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# --- 核心推理函数（滑动窗口）---
# MODIFIED: 函数现在返回一个包含两个预测图的元组
def test_single_case(net, image: np.ndarray, stride_xy: int, stride_z: int, patch_size: Tuple[int, int, int], num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    对单个3D图像进行滑动窗口推理。
    假设网络 `net` 返回一个包含两个输出的元组 `(output1, output2)`。
    
    返回:
        Tuple[np.ndarray, np.ndarray]: 两个输出对应的预测分割图 (pred_map1, pred_map2)。
    """
    patch_d, patch_h, patch_w = patch_size
    
    # 添加批次和通道维度，转换为张量，发送到GPU
    input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
    _, _, d, h, w = input_tensor.shape
    
    # MODIFIED: 为两个输出分别创建 score_map
    score_map1 = torch.zeros((1, num_classes, d, h, w), dtype=torch.float32)
    score_map2 = torch.zeros((1, num_classes, d, h, w), dtype=torch.float32)
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
                    
                    # MODIFIED: 接收模型的两个输出
                    # 假设您的网络返回一个元组 (main_output, auxiliary_output)
                    patch_pred1, patch_pred2 = net(patch)
                    
                    # MODIFIED: 分别累加两个输出的预测结果
                    score_map1[:, :, sz:sz + patch_d, sy:sy + patch_h, sx:sx + patch_w] += patch_pred1.cpu()
                    score_map2[:, :, sz:sz + patch_d, sy:sy + patch_h, sx:sx + patch_w] += patch_pred2.cpu()
                    count_map[:, :, sz:sz + patch_d, sy:sy + patch_h, sx:sx + patch_w] += 1.0
    
    count_map[count_map == 0] = 1.0
    
    # MODIFIED: 分别对两个 score_map 进行平均和 argmax
    score_map1 = score_map1 / count_map
    pred_map1 = torch.argmax(score_map1, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    
    score_map2 = score_map2 / count_map
    pred_map2 = torch.argmax(score_map2, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    
    return pred_map1, pred_map2

# --- MODIFIED: 更新辅助函数以包含 Jaccard 指数 ---
def log_final_stats(metrics_dict: Dict[str, List[float]], output_name: str):
    """计算并记录一组指标 (Dice, Jaccard, HD95) 的最终统计数据"""
    logging.info("\n" + "="*20 + f" 最终统计 ({output_name}) " + "="*20)
    
    # 检查是否有 'Dice' 键并且列表不为空
    if 'Dice' not in metrics_dict or not metrics_dict['Dice']:
        logging.warning(f"没有为 {output_name} 计算任何有效的指标。")
        return

    # --- 提取所有指标的数组 ---
    dice_array = np.array(metrics_dict['Dice'])
    # NEW: 提取 Jaccard
    jaccard_array = np.array(metrics_dict.get('Jaccard', [])) # 使用 .get() 保证健壮性
    hd_array = np.array(metrics_dict['HD95'])

    # --- Dice 统计 ---
    logging.info(f"[Dice]    均值 ± 标准差: {np.nanmean(dice_array):.4f} ± {np.nanstd(dice_array):.4f}")
    
    # --- NEW: Jaccard 统计 ---
    if jaccard_array.size > 0:
        logging.info(f"[Jaccard] 均值 ± 标准差: {np.nanmean(jaccard_array):.4f} ± {np.nanstd(jaccard_array):.4f}")

    # --- HD95 统计 ---
    # 过滤掉非法的 HD95 值 (nan)
    valid_hd_array = hd_array[~np.isnan(hd_array)]
    if valid_hd_array.size > 0:
        logging.info(f"[HD95]    均值 ± 标准差: {np.nanmean(valid_hd_array):.2f} ± {np.nanstd(valid_hd_array):.2f}")
    else:
        logging.info("[HD95]    所有案例均计算失败。")

def load_model(model, model_path, device):
    try:
        # 加载完整的 checkpoint 字典
        # weights_only=True 是一个安全措施，如果确认来源可信且需要加载非 tensor 数据，可以设为 False
        # 但对于 state_dict 通常是安全的。
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        logging.info(f"成功从 {model_path} 加载 checkpoint 文件。")

        # 检查 checkpoint 是否为字典
        if not isinstance(checkpoint, dict):
            raise TypeError(f"加载的 checkpoint 文件不是预期的字典格式，而是 {type(checkpoint)}。")

        # 优先选择 EMA 模型权重，其次是学生模型权重
        if 'ema_model_state_dict' in checkpoint:
            state_dict_to_load = checkpoint['ema_model_state_dict']
            logging.info("找到并选择 'ema_model_state_dict' 进行加载。")
        elif 'model_state_dict' in checkpoint:
            state_dict_to_load = checkpoint['model_state_dict']
            logging.info("未找到 EMA 权重，选择 'model_state_dict' 进行加载。")
        else:
            # 如果 checkpoint 中没有这两个键，可能是旧格式或者直接保存了 state_dict
            # 尝试将整个 checkpoint 作为 state_dict 加载，并发出警告
            logging.warning("在 checkpoint 中未找到 'ema_model_state_dict' 或 'model_state_dict'。"
                            "将尝试直接加载整个 checkpoint 内容作为 state_dict。")
            state_dict_to_load = checkpoint # 假设整个文件就是 state_dict

        # 检查提取出的 state_dict 是否为字典
        if not isinstance(state_dict_to_load, dict):
             raise TypeError(f"从 checkpoint 提取的 'state_dict_to_load' 不是字典，而是 {type(state_dict_to_load)}。")

        # 处理 'module.' 前缀（通常在 DataParallel 或 DistributedDataParallel 训练后出现）
        # 需要检查 state_dict_to_load 的键，而不是 checkpoint 的键
        needs_prefix_removal = any(key.startswith('module.') for key in state_dict_to_load.keys())

        if needs_prefix_removal:
            logging.info("检测到 'module.' 前缀，将进行移除。")
            new_state_dict = OrderedDict()
            for k, v in state_dict_to_load.items():
                if k.startswith('module.'):
                    name = k[7:]  # 移除 'module.'
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v # 如果某个键没有 module. 前缀，也保留
            state_dict_final = new_state_dict
        else:
            logging.info("未检测到 'module.' 前缀。")
            state_dict_final = state_dict_to_load

        # 加载处理后的状态字典到模型
        model.load_state_dict(state_dict_final)
        logging.info("成功将状态字典加载到模型。")

        # 将模型移动到指定设备
        model.to(device)
        logging.info(f"模型已移动到设备: {device}")

        # 将模型设置为评估模式（非常重要！）
        model.eval()
        logging.info("模型已设置为评估模式 (model.eval())。")

        return model

    except FileNotFoundError:
        logging.error(f"错误：模型文件未找到于 {model_path}")
        raise
    except Exception as e:
        logging.error(f"加载模型时发生错误: {e}")
        raise

# --- 主执行块 ---
if __name__ == '__main__':
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="在Amos数据集上测试VNet模型")
    parser.add_argument('--model_path', type=str, required=True, help='预训练VNet .pth模型文件路径')
    parser.add_argument('--amos_data_path', type=str, default='./datasets/Amos', help='Amos数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./result/VNet/Amos/Test', help='保存预测结果的目录')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'eval', 'train'], help='用于测试的Amos数据集分割')
    parser.add_argument('--speed', type=int, default=0, choices=[0, 1, 2], help='推理速度/步长设置')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='使用的GPU')
    parser.add_argument('--n_filters', type=int, default=16, help='VNet基础滤波器数量')
    parser.add_argument('--num_channels', type=int, default=1, help='VNet输入通道数')
    parser.add_argument('--metrics_log', type=str, default='./result/metrics_log.txt', help='评估指标日志文件路径')
    parser.add_argument('--save_images', action='store_true', help='是否保存预测结果图像')
    args = parser.parse_args()

    # ==================== 初始化设置 ====================
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    maybe_mkdir(args.output_dir)
    maybe_mkdir(os.path.dirname(args.metrics_log))

    setup_logging(args.metrics_log)
    logging.info("\n" + "="*40 + " 运行配置 " + "="*40)
    # ... (其余日志记录部分保持不变)
    for arg in vars(args):
        logging.info(f"{arg:20}: {getattr(args, arg)}")

    # ==================== 数据集配置 ====================
    try:
        amos_config = AmosConfig(save_dir=args.amos_data_path, patch_size=(80, 160, 160))
        patch_size = amos_config.patch_size
        num_classes = amos_config.num_classes
        stride_config = {0: (patch_size[1]//4, patch_size[0]//4), 1: (patch_size[1]//2, patch_size[0]//2), 2: (patch_size[1], patch_size[0])}
        stride_xy, stride_z = [max(1, x) for x in stride_config[args.speed]]
        logging.info(f"输入尺寸: {patch_size}, 类别数量: {num_classes}, 滑动窗口步长: XY={stride_xy}, Z={stride_z}")
    except Exception as e:
        logging.exception("数据集配置失败!")
        raise

    # ==================== 模型初始化 ====================
    try:
        # 确保你的VNet类能够返回两个输出
        model = Network(in_channels=1,num_classes=16).to(device=device) # 测试网络
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"模型总参数量: {total_params/1e6:.2f}M")
        
        model = load_model(model, args.model_path, device)
        logging.info(f"模型已加载: {args.model_path}")
            
        model.eval()
        logging.info("模型已设置为评估模式")
    except Exception as e:
        logging.exception("模型初始化失败!")
        raise

    # ==================== 数据加载 ====================
    temp_dataset = AmosDataset(split=args.split, config=amos_config)
    test_ids = temp_dataset.ids_list
    logging.info(f"加载数据集分割: {args.split}, 案例总数: {len(test_ids)}")

    # ==================== 推理流水线 ====================
    total_metrics_output1 = {'Dice': [], 'Jaccard': [], 'HD95': []}
    total_metrics_output2 = {'Dice': [], 'Jaccard': [], 'HD95': []}
    total_metrics_case_avg = {'Dice': [], 'Jaccard': [], 'HD95': []}

    logging.info("\n" + "="*40 + " 开始推理 " + "="*40)
    
    try:
        with torch.no_grad():
            progress_bar = tqdm(test_ids, desc=f"处理{args.split}数据集")
            for case_id in progress_bar:
                try:
                    # 加载数据
                    image, label = temp_dataset._load_sample(case_id)
                    
                    # 接收两个预测结果
                    pred1, pred2 = test_single_case(
                        model, image, 
                        stride_xy, stride_z,
                        patch_size, num_classes
                    )
                    
                    # 二值化预测和标签，为计算指标做准备
                    pred1_bin = (pred1 > 0).astype(np.uint8)
                    pred2_bin = (pred2 > 0).astype(np.uint8)
                    label_bin = (label > 0).astype(np.uint8)

                    # --- Output 1 指标计算 ---
                    dice1 = metric.binary.dc(pred1_bin, label_bin)
                    jc1 = metric.binary.jc(pred1_bin, label_bin) # NEW: 计算 Jaccard 1
                    try:
                        # 仅在 pred 和 label 都有前景时计算HD
                        if pred1_bin.sum() > 0 and label_bin.sum() > 0:
                            hd1 = metric.binary.hd95(pred1_bin, label_bin)
                        else:
                            hd1 = np.nan
                    except RuntimeError:
                        hd1 = np.nan
                        logging.warning(f"{case_id} (Output 1) HD95计算失败")
                    
                    # --- Output 2 指标计算 ---
                    dice2 = metric.binary.dc(pred2_bin, label_bin)
                    jc2 = metric.binary.jc(pred2_bin, label_bin) # NEW: 计算 Jaccard 2
                    try:
                        if pred2_bin.sum() > 0 and label_bin.sum() > 0:
                            hd2 = metric.binary.hd95(pred2_bin, label_bin)
                        else:
                            hd2 = np.nan
                    except RuntimeError:
                        hd2 = np.nan
                        logging.warning(f"{case_id} (Output 2) HD95计算失败")

                    # --- 计算当前病例的平均指标 ---
                    case_avg_dice = np.nanmean([dice1, dice2])
                    case_avg_jc = np.nanmean([jc1, jc2]) # NEW
                    case_avg_hd = np.nanmean([hd1, hd2])

                    # --- 分别记录所有指标 ---
                    total_metrics_output1['Dice'].append(dice1)
                    total_metrics_output1['Jaccard'].append(jc1) # NEW
                    total_metrics_output1['HD95'].append(hd1)
                    
                    total_metrics_output2['Dice'].append(dice2)
                    total_metrics_output2['Jaccard'].append(jc2) # NEW
                    total_metrics_output2['HD95'].append(hd2)

                    total_metrics_case_avg['Dice'].append(case_avg_dice)
                    total_metrics_case_avg['Jaccard'].append(case_avg_jc) # NEW
                    total_metrics_case_avg['HD95'].append(case_avg_hd)
                    
                    # --- MODIFIED: 修复日志记录的 f-string 语法 ---
                    
                    # 1. 准备好要显示的 HD95 字符串
                    hd1_str = f"{hd1:.2f}" if not np.isnan(hd1) else "NaN"
                    hd2_str = f"{hd2:.2f}" if not np.isnan(hd2) else "NaN"
                    case_avg_hd_str = f"{case_avg_hd:.2f}" if not np.isnan(case_avg_hd) else "NaN"
                    
                    # 2. 将准备好的字符串放入主 log_msg 中
                    log_msg = (
                        f"案例 {case_id} | "
                        f"Out1 [Dice:{dice1:.4f}, JC:{jc1:.4f}, HD95:{hd1_str}] | "
                        f"Out2 [Dice:{dice2:.4f}, JC:{jc2:.4f}, HD95:{hd2_str}] | "
                        f"平均 [Dice:{case_avg_dice:.4f}, JC:{case_avg_jc:.4f}, HD95:{case_avg_hd_str}]"
                    )
                    logging.info(log_msg)
                    
                    # MODIFIED: 如果需要保存图像，分别保存两个预测结果
                    if args.save_images:
                        # 保存 Output 1 的结果
                        sitk_img1 = sitk.GetImageFromArray(pred1.astype(np.uint8))
                        save_path1 = os.path.join(args.output_dir, f"{case_id}_pred1.nii.gz")
                        sitk.WriteImage(sitk_img1, save_path1)

                        # 保存 Output 2 的结果
                        sitk_img2 = sitk.GetImageFromArray(pred2.astype(np.uint8))
                        save_path2 = os.path.join(args.output_dir, f"{case_id}_pred2.nii.gz")
                        sitk.WriteImage(sitk_img2, save_path2)
                        
                except Exception as e:
                    logging.error(f"案例 {case_id} 处理失败: {str(e)}", exc_info=True)
                    continue
    except KeyboardInterrupt:
        logging.warning("用户中断执行!")
        raise

    finally:
        # ==================== 最终统计 ====================
        logging.info("\n" + "="*40 + " 推理完成，开始统计 " + "="*40)
        
        log_final_stats(total_metrics_output1, "Output 1 (Main)")
        log_final_stats(total_metrics_output2, "Output 2 (Aux)")
        log_final_stats(total_metrics_case_avg, "Per-Case Average (两个输出的平均)")

        # 记录显存使用
        if torch.cuda.is_available():
            logging.info(f"\n峰值显存使用: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        
        logging.info("\n测试流程完成!")