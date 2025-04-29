"""
V3版本的专用测试函数
即使没有什么差别也不建议其他版本使用
"""

import logging
import torch
import os
import argparse
from glob import glob
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # 启用Tensorboard
import h5py  # 用于读取和写入 HDF5 格式的文件
import math
import nibabel as nib  # 用于处理 NIfTI 格式的医学图像文件
import numpy as np
from medpy import metric  # 用于医学图像处理的评估指标函数
import torch
import torch.nn.functional as F
from tqdm import tqdm  # 进度条显示工具
from skimage.measure import label  # 用于标记连通区域
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 导入网络
from networks.SAM3D_VNet_SSL_V3 import Network
# 导入数据集
from dataloader.DataLoader_LA import LAHeart

# 设置随机种子
def set_seed(seed_value=42):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed_value)

# 设置日志记录
def setup_logging(log_file):
    """设置日志记录器，并确保日志目录存在"""
    # 获取日志文件的目录路径
    log_dir = os.path.dirname(log_file)

    # 如果日志目录不存在，则创建
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )

# 查找最新的模型检查点
def latest_checkpoint(path):
    """查找path中最新的文件"""
    list_of_files = glob(os.path.join(path, '*.pth'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# 加载模型
def load_model(model, model_path, device):
    """
    加载模型权重。优先加载 EMA 模型权重（如果存在）。
    处理 'module.' 前缀（用于 DataParallel/DistributedDataParallel 包装的模型）。
    并将模型设置为评估模式。

    Args:
        model (torch.nn.Module): 需要加载权重的模型实例。
        model_path (str): .pth 权重文件的路径。
        device (torch.device): 模型应该加载到的设备 (e.g., torch.device('cuda')).

    Returns:
        torch.nn.Module: 加载了权重的模型。
    """
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

# 获取图像分割结果的最大连通组件
def getLargestCC(segmentation):
    labels = label(segmentation)  # 对分割结果进行连通区域标记
    assert( labels.max() != 0 )  # 确保至少有一个连通组件
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1  # 找到最大连通区域
    return largestCC

# 用于解决一些错误
# def getLargestCC(segmentation):
#     labels = label(segmentation)
#     if labels.max() == 0:  # 没有连通组件（全背景）
#         return np.zeros_like(segmentation)  # 返回全0数组
#     largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
#     return largestCC.astype(np.uint8)

# 计算所有测试案例的平均Dice系数
def var_all_case(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, dataset_name="LA"):
    # 根据数据集名称，选择数据路径
    if dataset_name == "LA":
        with open('./datasets/LA/test.list', 'r') as f:
            image_list = f.readlines()  # 读取测试集列表
        image_list = ["./datasets/LA/data/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    elif dataset_name == "Pancreas_CT":
        with open('./datasets/Pancreas/test.list', 'r') as f:
            image_list = f.readlines()  # 读取测试集列表
        image_list = ["./data/Pancreas/data/" + item.replace('\n', '') + "_norm.h5" for item in image_list]

    loader = tqdm(image_list)  # 进度条
    total_dice = 0.0  # 累计Dice系数
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')  # 读取图像数据文件
        image = h5f['image'][:]  # 图像数据
        label = h5f['label'][:]  # 标签数据
        # 使用模型进行推断并计算预测结果和得分图
        prediction, score_map = test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        
        if np.sum(prediction) == 0:  # 如果没有预测结果
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)  # 计算Dice系数
        total_dice += dice  # 累加Dice系数

    # 计算平均Dice系数
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))  # 打印平均Dice系数
    return avg_dice

def test_all_case(model_name, num_outputs, model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=1, nms=0):
    loader = tqdm(image_list) if not metric_detail else image_list  # 如果需要详细信息则不使用进度条
    ith = 0  # 当前图像索引
    total_metric = 0.0  # 累计评估指标
    total_metric_average = 0.0  # 累计所有解码器的平均评估指标
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')  # 读取图像数据文件
        image = h5f['image'][:]  # 图像数据
        label = h5f['label'][:]  # 标签数据
        # print(f'image shape: {image.shape}, label shape: {label.shape}')

        if preproc_fn is not None:
            image = preproc_fn(image)  # 进行预处理（如果有）
        
        # 使用模型进行推断并获取分割结果和得分图
        prediction, score_map = test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        # print(f'prediction shape: {prediction.shape}, score_map shape: {score_map.shape}')

        if num_outputs > 1:  # 如果有多个输出解码器
            prediction_average, score_map_average = test_single_case_average_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        # if nms:  # 如果需要使用NMS进行后处理
        #     prediction = getLargestCC(prediction)  # 获取最大连通组件
        #     if num_outputs > 1:
        #         prediction_average = getLargestCC(prediction_average)

        # 针对报错修改了下面的内容
        if nms:
            # 对主解码器的预测结果处理
            if np.sum(prediction) > 0:
                prediction = getLargestCC(prediction)
            else:
                prediction = prediction  # 已经是全0，无需处理
            
            # 对平均解码器的预测结果处理
            if num_outputs > 1:
                if np.sum(prediction_average) > 0:
                    prediction_average = getLargestCC(prediction_average)
                else:
                    prediction_average = prediction_average

        # 计算每个图像的评估指标
        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)  # 如果没有预测结果，指标全为0
            if num_outputs > 1:
                single_metric_average = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])  # 计算评估指标（Dice, Jaccard, HD95, ASD）
            if num_outputs > 1:
                single_metric_average = calculate_metric_percase(prediction_average, label[:])
        
        # 记录每个图像的评估指标
        if metric_detail:
            logging.info(f"Image {ith:02d} - Dice: {single_metric[0]:.5f}, Jaccard: {single_metric[1]:.5f}, HD95: {single_metric[2]:.5f}, ASD: {single_metric[3]:.5f}")
            if num_outputs > 1:
                logging.info(f"Image {ith:02d} (average output) - Dice: {single_metric_average[0]:.5f}, Jaccard: {single_metric_average[1]:.5f}, HD95: {single_metric_average[2]:.5f}, ASD: {single_metric_average[3]:.5f}")

        # 累加评估指标
        total_metric += np.asarray(single_metric)
        if num_outputs > 1:
            total_metric_average += np.asarray(single_metric_average)

        # 保存预测结果和得分图
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred.nii.gz" % ith)
            nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
            if num_outputs > 1:
                nib.save(nib.Nifti1Image(prediction_average.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred_average.nii.gz" % ith)
                nib.save(nib.Nifti1Image(score_map_average[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores_average.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path +  "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path +  "%02d_gt.nii.gz" % ith)

        ith += 1

    # 计算所有图像的平均评估指标
    avg_metric = total_metric / len(image_list)
    logging.info(f"Average metric (decoder 1): {avg_metric}")
    if num_outputs > 1:
        avg_metric_average = total_metric_average / len(image_list)
        logging.info(f"Average metric (all decoders): {avg_metric_average}")

    # 记录平均评估指标到日志
    logging.info(f"Average metric of decoder 1: {avg_metric}")
    if num_outputs > 1:
        logging.info(f"Average metric of all decoders: {avg_metric_average}")

    return avg_metric

# 对单个图像进行推断并获取预测标签和得分图
def test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape  # 获取图像的尺寸（宽、高、深度）

    # 如果图像尺寸小于patch_size，则进行零填充
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0

    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2

    # 如果需要填充，则进行填充操作
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape  # 获取填充后的图像尺寸

    # 计算滑动窗口的步数
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)  # 初始化得分图
    cnt = np.zeros(image.shape).astype(np.float32)  # 初始化计数图

    # 对每个滑动窗口进行推断
    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])

                # 提取图像的一个patch并进行推断
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():  # 不计算梯度
                    y = model(test_patch)  # 模型推断
                    # print(f'y0 shape: {y[0].shape}')
                    # print(f'y1 shape: {y[1].shape}')

                    if len(y) > 1:
                        y = y[0]

                    y = F.softmax(y, dim=1)  # 使用softmax进行分类
                # print("Shape of y before numpy conversion:", y.shape)  # 检查维度
                y = y.cpu().data.numpy()  # 将结果转回numpy
                # print("Shape of y after numpy conversion:", y.shape)  # 检查 numpy 的维度
                y = y[0, 1, :, :, :]  # 获取分类结果（假设第二类是感兴趣的类别）

                # 更新得分图和计数图
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] = \
                    score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] = \
                    cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1

    # 归一化得分图
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = (score_map[0] > 0.5).astype(np.int64)  # 使用阈值0.5来生成二进制标签

    # 如果有填充，去掉填充部分
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    
    return label_map, score_map  # 返回预测标签和得分图

# 使用所有解码器的输出进行平均
def test_single_case_average_output(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape  # 获取图像尺寸

    # 填充图像
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0

    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2

    # 如果需要填充，则进行填充操作
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    # 计算滑动窗口步数
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    # 对每个滑动窗口进行推断
    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])

                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y_logit = net(test_patch)
                    num_outputs = len(y_logit)
                    y = torch.zeros(y_logit[0].shape).cuda()
                    for idx in range(num_outputs):
                        y += y_logit[idx]
                    y /= num_outputs  # 对所有输出解码器的结果进行平均

                y = y.cpu().data.numpy()
                y = y[0, 1, :, :, :]

                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] = \
                    score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] = \
                    cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1

    # 归一化得分图
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = (score_map[0] > 0.5).astype(np.int64)

    # 去掉填充部分
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]

    return label_map, score_map  # 返回平均输出的预测标签和得分图

# 计算每个图像的评估指标
def calculate_metric_percase_bk(pred, gt):
    dice = metric.binary.dc(pred, gt)  # 计算Dice系数
    jc = metric.binary.jc(pred, gt)  # 计算Jaccard系数
    hd = metric.binary.hd95(pred, gt)  # 计算95%的Hausdorff距离
    asd = metric.binary.asd(pred, gt)  # 计算平均表面距离（Average Symmetric Surface Distance）

    return dice, jc, hd, asd  # 返回四个评估指标

# 修改后的评估指标计算函数，预防了0的情况
def calculate_metric_percase(pred, gt):
    """
    计算单例的评估指标
    Args:
        pred: 预测结果 (必须是二值化的)
        gt: Ground Truth (必须是二值化的)
    Returns:
        dice, jc, hd, asd: 四个评估指标
    """
    # 确保输入是二值化的
    pred = (pred > 0.5).astype(int)
    gt = (gt > 0.5).astype(int)

    # 检查是否包含二值对象（至少有一个前景像素值为1）
    if pred.sum() == 0 or gt.sum() == 0:
        # 如果没有前景，无法计算 Hausdorff 距离和 ASD，设置为无效值
        dice = 0.0
        jc = 0.0
        hd = float('inf')  # 设置为无穷大表示无效
        asd = float('inf')  # 设置为无穷大表示无效
    else:
        # 计算评估指标
        dice = metric.binary.dc(pred, gt)  # 计算Dice系数
        jc = metric.binary.jc(pred, gt)  # 计算Jaccard系数
        hd = metric.binary.hd95(pred, gt)  # 计算95%的Hausdorff距离
        asd = metric.binary.asd(pred, gt)  # 计算平均表面距离

    return dice, jc, hd, asd  # 返回四个评估指标

# 主程序
if __name__ == '__main__':
    # 解析参数
    parser = argparse.ArgumentParser(description='Test a deep learning model on a given dataset and save checkpoint')
    parser.add_argument("--model_name", type=str, default='VNet', help='The name of the model')
    parser.add_argument("--model_load", type=str, default='./result/your_model/Pth/model_epoch_xx_checkpoint.pth', help='The path of the checkpoint')
    parser.add_argument("--log_path", type=str, default='./result/your_model/Test/running.log', help='The path of the test log')
    parser.add_argument("--patch_size", type=str, default='(112, 112, 80)', help='The resolution of test image size (tuple as string)')
    parser.add_argument("--test_save_path", type=str, default=None, help="The path to save segmentation results")
    parser.add_argument("--root_path", type=str, default='./datasets/LA', help='The path of the dataset')
    parser.add_argument("--num_classes", type=int, default=2, help='The number of classes')
    parser.add_argument("--num_outputs", type=int, default=1, help='The number of outputs of the model')
    option = parser.parse_args()

     # 设置日志
    setup_logging(option.log_path)
    logging.info(f"Log file will be saved at: {option.log_path}")

    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info(f"Using CPU for computations")

    # 创建模型实例
    logging.info(f"Creating model: {option.model_name}")
    # model = Network().to(device=device) # V1原本
    model = Network(pretrain_weight_path="./result/VNet/LA/Pth/best.pth",encoder_depth=8).to(device=device)

    # 加载模型
    logging.info(f"Loading model weights from: {option.model_load}")
    model = load_model(model, option.model_load, device)
    model.eval()  # 进入评估模式
    
    # 设置数据集路径
    test_list_path = f"{option.root_path}/test.list"
    logging.info(f"Loading test list from: {test_list_path}")
    try:
        with open(test_list_path, 'r') as f:
            image_list = f.readlines()
        logging.info(f"Successfully loaded {len(image_list)} image paths from test list.")
    except Exception as e:
        logging.error(f"Error reading test list: {e}")
    
    # 构建完整的图像路径
    image_list = [f"{option.root_path}/data/{item.strip()}/mri_norm2.h5" for item in image_list]
    logging.info(f"Constructed full image paths for {len(image_list)} images.")
    
    # 评估
    logging.info(f"Starting the performance evaluation for {option.model_name}.")
    avg_metric = test_all_case(
        model_name=option.model_name, 
        num_outputs=option.num_outputs,
        model=model, 
        image_list=image_list,
        num_classes=option.num_classes, 
        patch_size=tuple(map(int, option.patch_size.strip('()').split(','))),
        stride_xy=18, stride_z=4, 
        save_result=False, 
        test_save_path=option.test_save_path, 
        preproc_fn=None, 
        metric_detail=1, 
        nms=1  # 采用NMS后处理
    )

    logging.info(f"Test results: {avg_metric}")