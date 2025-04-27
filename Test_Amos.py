# Test_Amos.py
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import csv
from datetime import datetime
import SimpleITK as sitk
from tqdm import tqdm
from medpy import metric
from typing import Tuple, Dict

# 导入VNet模型和数据集类
from networks.VNet import VNet

# 导入Amos数据集
from dataloader.DataLoader_Amos import AmosConfig, AmosDataset

# 如果目录不存在则创建
def maybe_mkdir(path):
    os.makedirs(path, exist_ok=True)

# --- 核心推理函数（滑动窗口） ---
def test_single_case(net, image: np.ndarray, stride_xy: int, stride_z: int, patch_size: Tuple[int, int, int], num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    在单个3D图像上执行滑动窗口推理
    
    参数:
        net: 神经网络模型（已处于评估模式并在CUDA上）
        image: 输入图像，NumPy数组 (D, H, W)，假设C=1
        stride_xy: 高度和宽度维度的步长
        stride_z: 深度维度的步长
        patch_size: 输入补丁的尺寸 (patch_d, patch_h, patch_w)
        num_classes: 输出类别数
    
    返回:
        包含以下内容的元组:
        - pred_map: 最终分割图（按类别argmax），NumPy数组 (D, H, W)
        - score_map: 原始概率/逻辑图，NumPy数组 (C, D, H, W)
    """
    patch_d, patch_h, patch_w = patch_size
    img_d, img_h, img_w = image.shape
    
    # 添加批次和通道维度，转换为张量，发送到GPU
    # VNet期望的输入形状：(B, C, D, H, W)
    input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
    _, _, d, h, w = input_tensor.shape
    
    # 在CPU上初始化分数图和计数图以节省GPU内存
    score_map = torch.zeros((1, num_classes, d, h, w), dtype=torch.float32)  # 存储logits或概率
    count_map = torch.zeros((1, 1, d, h, w), dtype=torch.float32)
    
    # 计算步数
    steps_z = int(np.ceil((d - patch_d) / stride_z)) + 1
    steps_y = int(np.ceil((h - patch_h) / stride_xy)) + 1
    steps_x = int(np.ceil((w - patch_w) / stride_xy)) + 1
    
    print(f"图像形状: {image.shape}, 补丁尺寸: {patch_size}")
    print(f"步长: z={stride_z}, xy={stride_xy}")
    print(f"步数: z={steps_z}, y={steps_y}, x={steps_x}")
    
    with torch.no_grad():
        for iz in range(steps_z):
            sz = min(stride_z * iz, d - patch_d)  # 计算起始z坐标
            for iy in range(steps_y):
                sy = min(stride_xy * iy, h - patch_h)  # 计算起始y坐标
                for ix in range(steps_x):
                    sx = min(stride_xy * ix, w - patch_w)  # 计算起始x坐标
                    # 提取补丁
                    patch = input_tensor[:, :, sz:sz + patch_d, sy:sy + patch_h, sx:sx + patch_w]
                    # 运行推理
                    patch_pred = net(patch)  # 输出形状：(B, num_classes, patch_d, patch_h, patch_w)
                    # 将预测添加到分数图并增加计数图
                    score_map[:, :, sz:sz + patch_d, sy:sy + patch_h, sx:sx + patch_w] += patch_pred.cpu()
                    count_map[:, :, sz:sz + patch_d, sy:sy + patch_h, sx:sx + patch_w] += 1.0
    
    # 避免除以零
    count_map[count_map == 0] = 1.0
    # 在重叠区域平均分数
    score_map = score_map / count_map
    
    # 如果需要，应用Softmax获取概率（可选，取决于score_map的使用方式）
    # score_map_prob = F.softmax(score_map, dim=1)
    
    # 通过argmax获取最终预测
    pred_map = torch.argmax(score_map, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    score_map_np = score_map.squeeze(0).cpu().numpy()  # 形状：(C, D, H, W)
    
    print(f"预测图形状: {pred_map.shape}")  # 应为 (D, H, W)
    print(f"分数图形状: {score_map_np.shape}")  # 应为 (num_classes, D, H, W)
    return pred_map, score_map_np

def calculate_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int) -> dict:
    """
    计算Dice系数和HD95距离
    """
    metrics = {'Dice': 0.0, 'HD95': 0.0}
    
    # 仅计算前景类别（排除背景0）
    for cls in range(1, num_classes):
        pred_cls = (pred == cls).astype(np.uint8)
        target_cls = (target == cls).astype(np.uint8)
        
        if np.sum(target_cls) == 0:
            continue
            
        # 计算Dice
        dice = metric.binary.dc(pred_cls, target_cls)
        metrics['Dice'] += dice
        
        # 计算HD95
        try:
            hd = metric.binary.hd95(pred_cls, target_cls)
            metrics['HD95'] += hd
        except RuntimeError:
            hd = np.nan
            
    # 平均指标
    valid_classes = num_classes - 1  # 排除背景
    metrics['Dice'] /= valid_classes
    metrics['HD95'] /= valid_classes
    
    return metrics

def save_metrics_to_log(metrics_log: str, case_metrics: dict, header: bool = False):
    """保存指标到CSV日志文件"""
    fieldnames = ['CaseID', 'Dice', 'HD95', 'Timestamp']
    mode = 'w' if header else 'a'
    
    with open(metrics_log, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if header:
            writer.writeheader()
        writer.writerow({
            'CaseID': case_metrics['id'],
            'Dice': case_metrics['Dice'],
            'HD95': case_metrics['HD95'],
            'Timestamp': datetime.now().isoformat()
        })

# --- 主执行块 ---
# --- 主执行块 ---
if __name__ == '__main__':
    # 参数解析
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
    parser.add_argument('--metrics_log', type=str, default='./result/metrics_log.csv',
                      help='评估指标日志文件路径')
    parser.add_argument('--save_images', action='store_true',
                      help='是否保存预测结果图像')
    
    args = parser.parse_args()
    
    # 初始化设置
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maybe_mkdir(args.output_dir)
    
    # 数据集配置（关键修改点1：设置全尺寸）
    amos_config = AmosConfig(
        save_dir=args.amos_data_path,
        patch_size=(80, 160, 160)  # 全尺寸设置
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
    
    # 模型初始化
    model = VNet(
        n_channels=args.num_channels,
        n_classes=num_classes,
        n_filters=args.n_filters,
        normalization='batchnorm',
        has_dropout=False
    ).to(device)
    
    # 加载模型权重
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    else:
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    model.eval()

    # 数据加载
    temp_dataset = AmosDataset(split=args.split, config=amos_config)
    test_ids = temp_dataset.ids_list
    
    # 指标记录初始化
    metrics_header = ['CaseID', 'Dice', 'HD95', 'Timestamp']
    if not os.path.exists(args.metrics_log):
        with open(args.metrics_log, 'w') as f:
            csv.writer(f).writerow(metrics_header)
    
    total_metrics = {'Dice': [], 'HD95': []}

    # 推理循环
    with torch.no_grad():
        for case_id in tqdm(test_ids, desc=f"处理{args.split}数据集"):
            try:
                # 加载数据
                image, label = temp_dataset._load_sample(case_id)
                
                # 执行推理
                pred, _ = test_single_case(
                    model, image, 
                    stride_xy, stride_z,
                    patch_size, num_classes
                )
                
                # 计算指标
                dice = metric.binary.dc(pred > 0, label > 0)  # 全局Dice
                hd = metric.binary.hd95(pred > 0, label > 0)
                
                # 记录指标
                case_metrics = {
                    'CaseID': case_id,
                    'Dice': round(dice, 4),
                    'HD95': round(hd, 2),
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(args.metrics_log, 'a') as f:
                    csv.DictWriter(f, metrics_header).writerow(case_metrics)
                
                total_metrics['Dice'].append(dice)
                total_metrics['HD95'].append(hd)
                
                # 保存预测结果
                if args.save_images:
                    sitk_img = sitk.GetImageFromArray(pred.astype(np.uint8))
                    sitk.WriteImage(sitk_img, os.path.join(args.output_dir, f"{case_id}_pred.nii.gz"))
            except Exception as e:
                print(f"处理{case_id}时出错: {str(e)}")
                continue

    # 输出最终统计结果（关键修改点2：修复类型错误）
    print("\n" + "="*40)
    print(f"全局评估结果（共{len(total_metrics['Dice'])}例）:")  # 正确访问列表长度
    print(f"平均Dice系数: {np.nanmean(total_metrics['Dice']):.4f} ± {np.nanstd(total_metrics['Dice']):.4f}")
    print(f"平均HD95距离: {np.nanmean(total_metrics['HD95']):.2f} ± {np.nanstd(total_metrics['HD95']):.2f}mm")
    print(f"详细指标已保存至: {args.metrics_log}")
    print("="*40)