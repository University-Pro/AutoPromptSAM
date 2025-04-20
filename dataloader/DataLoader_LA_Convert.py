#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
h5_to_nii.py

功能：将 LA 数据集中 data/<sample>/mri_norm2.h5 中的 image 和 label
保存为 nii.gz，方便可视化。

用法示例：
    python h5_to_nii.py \
        --base_dir ./datasets/LA \
        --split train \
        --num 20 \
        --output_dir ./datasets/LA/nii_data

将只转换前 20 个 train 样本，输出到 ./datasets/LA/nii_data/train 下。
"""

import os
import argparse
import h5py
import numpy as np
import nibabel as nib
from tqdm import tqdm

def convert_h5_to_nii(base_dir: str,
                      split: str = 'train',
                      num: int = None,
                      output_dir: str = None):
    """
    将 H5 格式的 3D 图像和标签保存为 nii.gz。

    参数:
        base_dir:     根目录，包含 train.list、test.list、data/ 结构
        split:        'train' 或 'test'，对应 base_dir/train.list
        num:          转换的样本数，None 表示全部
        output_dir:   输出主目录，默认 base_dir/nii_data/<split>
    返回:
        最终输出目录路径
    """
    # 准备输出目录
    if output_dir is None:
        output_dir = os.path.join(base_dir, 'nii_data', split)
    os.makedirs(output_dir, exist_ok=True)

    # 读取样本列表
    list_path = os.path.join(base_dir, f"{split}.list")
    with open(list_path, 'r') as f:
        samples = [x.strip() for x in f if x.strip()]
    if num is not None:
        samples = samples[:num]

    print(f"开始转换 {split} 集合，共 {len(samples)} 个样本...")
    for name in tqdm(samples, ncols=80):
        # 原 H5 路径
        h5_path = os.path.join(base_dir, 'data', name, 'mri_norm2.h5')
        if not os.path.isfile(h5_path):
            tqdm.write(f"[警告] 找不到 {h5_path}，跳过")
            continue

        # 每个样本创建子目录
        sample_out = os.path.join(output_dir, name)
        os.makedirs(sample_out, exist_ok=True)

        # 载入 H5
        with h5py.File(h5_path, 'r') as h5f:
            img = h5f['image'][()]   # ndarray, e.g. (H, W, D) 或 (C, H, W, D)
            lab = h5f['label'][()]

        # 去掉单通道维度
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]
        if lab.ndim == 4 and lab.shape[0] == 1:
            lab = lab[0]

        # 强制类型
        img = img.astype(np.float32)
        lab = lab.astype(np.int16)

        # NIfTI 需要一个 4x4 仿射矩阵，这里先用单位矩阵
        affine = np.eye(4, dtype=np.float32)

        # 保存
        img_nii_path = os.path.join(sample_out, 'image.nii.gz')
        lab_nii_path = os.path.join(sample_out, 'label.nii.gz')
        nib.save(nib.Nifti1Image(img, affine), img_nii_path)
        nib.save(nib.Nifti1Image(lab, affine), lab_nii_path)

    print(f"全部转换完成，结果保存在: {output_dir}")
    return output_dir

def parse_args():
    parser = argparse.ArgumentParser(
        description="将 LA 数据集的 H5 转换为 NIfTI (.nii.gz) 格式")
    parser.add_argument('--base_dir',   type=str, required=True,
                        help="LA 数据集根目录，包含 train.list、test.list、data/ 等")
    parser.add_argument('--split',      type=str, default='train',
                        choices=['train','test'], help="要转换的子集")
    parser.add_argument('--num',        type=int, default=None,
                        help="最多转换多少个样本，默认全部")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="输出目录，默认 base_dir/nii_data/<split>")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    convert_h5_to_nii(
        base_dir   = args.base_dir,
        split      = args.split,
        num        = args.num,
        output_dir = args.output_dir
    )