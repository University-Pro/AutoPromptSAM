# Test_Amos.py
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
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

# --- 主执行块 ---
if __name__ == '__main__':
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
    # 如果VNet参数未固定或未保存在检查点中，则添加参数
    parser.add_argument('--n_filters', type=int, default=16, help='VNet基础滤波器数量')
    parser.add_argument('--num_channels', type=int, default=1, help='VNet输入通道数')
    
    args = parser.parse_args()
    
    # 设置GPU环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # --- 配置 ---
    # 使用AmosConfig获取数据集特定参数
    amos_config = AmosConfig(save_dir=args.amos_data_path)
    
    # 可能需要根据VNet训练方式或内存限制调整patch_size
    # 这里使用AmosConfig的默认值，但要确保与VNet兼容
    patch_size = amos_config.patch_size
    num_classes = amos_config.num_classes  # 从Amos配置获取类别数
    
    # 根据速度参数定义步长（类似原始脚本）
    stride_dict = {
        # stride_xy, stride_z（根据patch_size和期望重叠调整）
        0: (patch_size[1] // 4, patch_size[0] // 4),  # 小步长，更多重叠（更慢）
        1: (patch_size[1] // 2, patch_size[0] // 2),  # 中步长
        2: (patch_size[1], patch_size[0]),            # 大步长，更少重叠（更快）- 可能效果较差
    }
    
    # 确保步长至少为1
    stride_xy = max(1, stride_dict[args.speed][0])
    stride_z = max(1, stride_dict[args.speed][1])
    
    # 创建输出目录
    test_save_path = args.output_dir
    maybe_mkdir(test_save_path)
    print(f"预测结果将保存至: {test_save_path}")
    
    # --- 模型初始化和加载 ---
    print("正在初始化VNet模型...")
    # 确保VNet参数与加载的检查点匹配
    model = VNet(
        n_channels=args.num_channels,  # 灰度医学图像应为1
        n_classes=num_classes,
        n_filters=args.n_filters,
        normalization='batchnorm',  # 或根据训练使用'groupnorm'
        has_dropout=False           # 推理时通常为False
    ).cuda()
    
    # 如果VNet定义中没有num_classes属性则添加（用于虚拟类）
    if not hasattr(model, 'num_classes'):
        model.num_classes = num_classes

    print(f"正在从 {args.model_path} 加载模型权重...")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"在 {args.model_path} 找不到模型检查点")
    try:
        # 加载状态字典
        checkpoint = torch.load(args.model_path, map_location='cuda')
        # 尝试常见状态字典键名，必要时调整
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
             model.load_state_dict(checkpoint['model_state_dict'])
        elif 'net' in checkpoint:
             model.load_state_dict(checkpoint['net'])
        elif 'A' in checkpoint and 'B' not in checkpoint:  # 处理双模型设置中的单个模型保存
             model.load_state_dict(checkpoint['A'])
        else:
            # 假设检查点文件直接包含状态字典
            model.load_state_dict(checkpoint)
        print("模型权重加载成功。")
    except Exception as e:
        print(f"加载状态字典时出错: {e}")
        print("请确保模型架构与检查点匹配，且检查点文件有效。")
        exit(1)
    
    model.eval()  # 设置模型为评估模式
    
    # --- 数据集和测试循环 ---
    print(f"正在从 {args.amos_data_path} 加载Amos数据集...")
    # 如果手动加载样本，不需要完整的Dataset对象实例
    # 但需要ID列表和加载逻辑
    # 实例化虚拟数据集以使用其_read_list和_load_sample方法
    # 注意：_load_sample需要实例具有self.config
    temp_dataset = AmosDataset(split=args.split, config=amos_config)
    test_ids_list = temp_dataset.ids_list  # 获取指定分割的ID列表
    
    print(f"开始在 '{args.split}' 分割的 {len(test_ids_list)} 个病例上进行推理...")
    with torch.no_grad():  # 禁用梯度计算以进行推理
        for data_id in tqdm(test_ids_list, desc=f"在Amos {args.split}分割上测试"):
            try:
                # 使用AmosDataset的逻辑加载和预处理图像
                # 确保加载/归一化方式与VNet训练时一致
                image, _ = temp_dataset._load_sample(data_id)  # 返回归一化后的图像 (D, H, W)
                print(f"\n正在处理病例: {data_id}, 图像形状: {image.shape}")
                
                # 使用滑动窗口函数执行推理
                pred_map, score_map = test_single_case(
                    model,
                    image,
                    stride_xy=stride_xy,
                    stride_z=stride_z,
                    patch_size=patch_size,
                    num_classes=num_classes
                )  # pred_map 形状: (D, H, W)
                
                # 将预测图保存为NIfTI文件
                output_filename = os.path.join(test_save_path, f"{data_id}_pred.nii.gz")
                
                sitk_out = sitk.GetImageFromArray(pred_map.astype(np.uint8))
                
                sitk.WriteImage(sitk_out, output_filename)

            except FileNotFoundError as e:
                print(f"处理 {data_id} 时出错：文件未找到 - {e}。已跳过。")
            except Exception as e:
                print(f"处理 {data_id} 时出错：{e}。已跳过。")
                # import traceback
                # traceback.print_exc()  # 取消注释以查看详细堆栈跟踪
    
    print("\n推理完成。")
    print(f"预测结果保存在: {test_save_path}")