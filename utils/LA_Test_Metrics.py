"""
LA数据集的评估指标计算相关的函数
"""
import h5py  # 用于读取和写入 HDF5 格式的文件
import math
import nibabel as nib  # 用于处理 NIfTI 格式的医学图像文件
import numpy as np
from medpy import metric  # 用于医学图像处理的评估指标函数
import torch
import torch.nn.functional as F
from tqdm import tqdm  # 进度条显示工具
from skimage.measure import label  # 用于标记连通区域
import logging

# 获取图像分割结果的最大连通组件
def getLargestCC(segmentation):
    labels = label(segmentation)  # 对分割结果进行连通区域标记
    assert( labels.max() != 0 )  # 确保至少有一个连通组件
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1  # 找到最大连通区域
    return largestCC

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
        
        if num_outputs > 1:  # 如果有多个输出解码器
            prediction_average, score_map_average = test_single_case_average_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        if nms:  # 如果需要使用NMS进行后处理
            prediction = getLargestCC(prediction)  # 获取最大连通组件
            if num_outputs > 1:
                prediction_average = getLargestCC(prediction_average)

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
                    # print(f'y shape: {y.shape}')

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
