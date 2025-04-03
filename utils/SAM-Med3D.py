# -*- encoding: utf-8 -*-
'''
@File    :   infer_with_medim.py
@Time    :   2024/09/08 11:31:02
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   使用MedIM进行推理的示例代码
'''

import medim
import torch
import numpy as np
import torch.nn.functional as F
import torchio as tio
import os.path as osp
import os
from torchio.data.io import sitk_to_nib
import SimpleITK as sitk


def data_preprocess(img_path, gt_path, category_index):
    """
    数据预处理函数
    
    参数:
    - img_path: 图像路径
    - gt_path: 标签路径
    - category_index: 目标类别的索引
    
    返回:
    - 预处理后的图像、标签和元信息
    """
    target_img_path = osp.join(
        osp.dirname(img_path),
        osp.basename(img_path).replace(".nii.gz", "_resampled.nii.gz"))
    target_gt_path = osp.join(
        osp.dirname(gt_path),
        osp.basename(gt_path).replace(".nii.gz", "_resampled.nii.gz"))
    resample_nii(img_path, target_img_path)
    resample_nii(gt_path,
                 target_gt_path,
                 n=category_index,
                 reference_image=tio.ScalarImage(target_img_path),
                 mode="nearest")
    roi_image, roi_label, meta_info = read_data_from_nii(
        target_img_path, target_gt_path)
    
    # 在返回前添加标签验证
    if torch.sum(roi_label) == 0:
        raise ValueError(f"标签数据全为0，请检查：\n"
                         f"1. 输入路径是否正确（当前标签路径：{gt_path}）\n"
                         f"2. category_index是否正确（当前值：{category_index}）")
    
    return roi_image, roi_label, meta_info

def random_sample_next_click(prev_mask, gt_mask):
    """
    从ground-truth mask和之前的seg mask中随机采样一个点击点

    参数:
        prev_mask: (torch.Tensor) [H,W,D] SAM-Med3D预测的之前mask
        gt_mask: (torch.Tensor) [H,W,D] 图像的ground-truth mask
    """
    prev_mask = prev_mask > 0
    true_masks = gt_mask > 0

    # 修改错误提示信息更友好
    if (not true_masks.any()):
        raise ValueError(
            "标签数据验证失败！可能原因：\n"
            "1. 输入的标签文件不正确\n"
            "2. 预处理时category_index设置错误\n"
            "3. 原始标签数据本身存在问题\n"
            f"当前标签路径：{gt_path}\n"  # 需要从上层传递gt_path参数
            f"检测到的标签值范围：[{torch.min(gt_mask)} ~ {torch.max(gt_mask)}]"
        )
    fn_masks = torch.logical_and(true_masks, torch.logical_not(prev_mask))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), prev_mask)

    to_point_mask = torch.logical_or(fn_masks, fp_masks)

    all_points = torch.argwhere(to_point_mask)
    point = all_points[np.random.randint(len(all_points))]

    if fn_masks[point[0], point[1], point[2]]:
        is_positive = True
    else:
        is_positive = False

    sampled_point = point.clone().detach().reshape(1, 1, 3)
    sampled_label = torch.tensor([
        int(is_positive),
    ]).reshape(1, 1)

    return sampled_point, sampled_label


def sam_model_infer(model,
                    roi_image,
                    prompt_generator=random_sample_next_click,
                    roi_gt=None,
                    prev_low_res_mask=None):
    '''
    SAM-Med3D的推理函数，输入提示点及其标签（每个点的正/负）

    # roi_image: (torch.Tensor) 裁剪后的图像，形状 [1,1,128,128,128]
    # prompt_points_and_labels: (Tuple(torch.Tensor, torch.Tensor))
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device", device)
    model = model.to(device)

    with torch.no_grad():
        input_tensor = roi_image.to(device)
        image_embeddings = model.image_encoder(input_tensor)

        points_coords, points_labels = torch.zeros(1, 0,
                                                   3).to(device), torch.zeros(
                                                       1, 0).to(device)
        new_points_co, new_points_la = torch.Tensor(
            [[[64, 64, 64]]]).to(device), torch.Tensor([[1]]).to(torch.int64)
        if (roi_gt is not None):
            prev_low_res_mask = prev_low_res_mask if (
                prev_low_res_mask is not None) else torch.zeros(
                    1, 1, roi_image.shape[2] // 4, roi_image.shape[3] //
                    4, roi_image.shape[4] // 4)
            new_points_co, new_points_la = prompt_generator(
                torch.zeros_like(roi_image)[0, 0], roi_gt[0, 0])
            new_points_co, new_points_la = new_points_co.to(
                device), new_points_la.to(device)
        points_coords = torch.cat([points_coords, new_points_co], dim=1)
        points_labels = torch.cat([points_labels, new_points_la], dim=1)

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=[points_coords, points_labels],
            boxes=None,  # we currently not support bbox prompt
            masks=prev_low_res_mask.to(device),
            # masks=None,
        )

        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embeddings,  # (1, 384, 8, 8, 8)
            image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 384, 8, 8, 8)
            sparse_prompt_embeddings=sparse_embeddings,  # (1, 2, 384)
            dense_prompt_embeddings=dense_embeddings,  # (1, 384, 8, 8, 8)
        )

        prev_mask = F.interpolate(low_res_masks,
                                  size=roi_image.shape[-3:],
                                  mode='trilinear',
                                  align_corners=False)

    # convert prob to mask
    medsam_seg_prob = torch.sigmoid(prev_mask)  # (1, 1, 64, 64, 64)
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg_mask = (medsam_seg_prob > 0.5).astype(np.uint8)

    return medsam_seg_mask


def resample_nii(input_path: str,
                 output_path: str,
                 target_spacing: tuple = (1.5, 1.5, 1.5),
                 n=None,
                 reference_image=None,
                 mode="linear"):
    """
    使用torchio将nii.gz文件重采样到指定spacing

    参数:
    - input_path: 输入.nii.gz文件的路径
    - output_path: 保存重采样后的.nii.gz文件的路径
    - target_spacing: 重采样的目标spacing，默认为(1.5, 1.5, 1.5)
    """
    # Load the nii.gz file using torchio
    subject = tio.Subject(img=tio.ScalarImage(input_path))
    resampler = tio.Resample(target=target_spacing, image_interpolation=mode)
    resampled_subject = resampler(subject)

    if (n != None):
        image = resampled_subject.img
        tensor_data = image.data
        if (isinstance(n, int)):
            n = [n]
        for ni in n:
            tensor_data[tensor_data == ni] = -1
        tensor_data[tensor_data != -1] = 0
        tensor_data[tensor_data != 0] = 1
        save_image = tio.ScalarImage(tensor=tensor_data, affine=image.affine)
        reference_size = reference_image.shape[
            1:]  # omitting the channel dimension
        cropper_or_padder = tio.CropOrPad(reference_size)
        save_image = cropper_or_padder(save_image)
    else:
        save_image = resampled_subject.img

    save_image.save(output_path)


def read_data_from_nii(img_path, gt_path):
    """
    从nii文件中读取数据并进行预处理
    
    返回:
    - 裁剪后的图像和标签
    - 元信息（原始图像信息、裁剪参数等）
    """
    sitk_image = sitk.ReadImage(img_path)
    sitk_label = sitk.ReadImage(gt_path)

    if sitk_image.GetOrigin() != sitk_label.GetOrigin():
        sitk_image.SetOrigin(sitk_label.GetOrigin())
    if sitk_image.GetDirection() != sitk_label.GetDirection():
        sitk_image.SetDirection(sitk_label.GetDirection())

    sitk_image_arr, _ = sitk_to_nib(sitk_image)
    sitk_label_arr, _ = sitk_to_nib(sitk_label)

    subject = tio.Subject(
        image=tio.ScalarImage(tensor=sitk_image_arr),
        label=tio.LabelMap(tensor=sitk_label_arr),
    )
    crop_transform = tio.CropOrPad(mask_name='label',
                                   target_shape=(128, 128, 128))
    padding_params, cropping_params = crop_transform.compute_crop_or_pad(
        subject)
    if (cropping_params is None): cropping_params = (0, 0, 0, 0, 0, 0)
    if (padding_params is None): padding_params = (0, 0, 0, 0, 0, 0)

    infer_transform = tio.Compose([
        crop_transform,
        tio.ZNormalization(masking_method=lambda x: x > 0),
    ])
    subject_roi = infer_transform(subject)

    img3D_roi, gt3D_roi = subject_roi.image.data.clone().detach().unsqueeze(
        1), subject_roi.label.data.clone().detach().unsqueeze(1)
    ori_roi_offset = (
        cropping_params[0],
        cropping_params[0] + 128 - padding_params[0] - padding_params[1],
        cropping_params[2],
        cropping_params[2] + 128 - padding_params[2] - padding_params[3],
        cropping_params[4],
        cropping_params[4] + 128 - padding_params[4] - padding_params[5],
    )

    meta_info = {
        "image_path": img_path,
        "image_shape": sitk_image_arr.shape[1:],
        "origin": sitk_label.GetOrigin(),
        "direction": sitk_label.GetDirection(),
        "spacing": sitk_label.GetSpacing(),
        "padding_params": padding_params,
        "cropping_params": cropping_params,
        "ori_roi": ori_roi_offset,
    }
    return (
        img3D_roi,
        gt3D_roi,
        meta_info,
    )


def save_numpy_to_nifti(in_arr: np.array, out_path, meta_info):
    """
    将numpy数组保存为nifti文件
    
    参数:
    - in_arr: 输入numpy数组
    - out_path: 输出文件路径
    - meta_info: 元信息（原点、方向、spacing等）
    """
    # torchio turn 1xHxWxD -> DxWxH
    # so we need to squeeze and transpose back to HxWxD
    ori_arr = np.transpose(in_arr.squeeze(), (2, 1, 0))
    out = sitk.GetImageFromArray(ori_arr)
    sitk_meta_translator = lambda x: [float(i) for i in x]
    out.SetOrigin(sitk_meta_translator(meta_info["origin"]))
    out.SetDirection(sitk_meta_translator(meta_info["direction"]))
    out.SetSpacing(sitk_meta_translator(meta_info["spacing"]))
    sitk.WriteImage(out, out_path)


def data_preprocess(img_path, gt_path, category_index):
    """
    数据预处理函数
    
    参数:
    - img_path: 图像路径
    - gt_path: 标签路径
    - category_index: 目标类别的索引
    
    返回:
    - 预处理后的图像、标签和元信息
    """
    target_img_path = osp.join(
        osp.dirname(img_path),
        osp.basename(img_path).replace(".nii.gz", "_resampled.nii.gz"))
    target_gt_path = osp.join(
        osp.dirname(gt_path),
        osp.basename(gt_path).replace(".nii.gz", "_resampled.nii.gz"))
    resample_nii(img_path, target_img_path)
    resample_nii(gt_path,
                 target_gt_path,
                 n=category_index,
                 reference_image=tio.ScalarImage(target_img_path),
                 mode="nearest")
    roi_image, roi_label, meta_info = read_data_from_nii(
        target_img_path, target_gt_path)
    return roi_image, roi_label, meta_info


def data_postprocess(roi_pred, meta_info, output_path, ori_img_path):
    """
    数据后处理函数
    
    参数:
    - roi_pred: 预测结果
    - meta_info: 元信息
    - output_path: 输出路径
    - ori_img_path: 原始图像路径
    """
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    pred3D_full = np.zeros(meta_info["image_shape"])
    ori_roi = meta_info["ori_roi"]
    pred3D_full[ori_roi[0]:ori_roi[1], ori_roi[2]:ori_roi[3],
                ori_roi[4]:ori_roi[5]] = roi_pred

    sitk_image = sitk.ReadImage(ori_img_path)
    ori_meta_info = {
        "image_path": ori_img_path,
        "image_shape": sitk_image.GetSize(),
        "origin": sitk_image.GetOrigin(),
        "direction": sitk_image.GetDirection(),
        "spacing": sitk_image.GetSpacing(),
    }
    pred3D_full_ori = F.interpolate(
        torch.Tensor(pred3D_full)[None][None],
        size=ori_meta_info["image_shape"],
        mode='nearest').cpu().numpy().squeeze()
    save_numpy_to_nifti(pred3D_full_ori, output_path, meta_info)


if __name__ == "__main__":
    ''' 1. 读取并预处理输入数据 '''
    img_path = "./datasets/LA/nii_data/0RZDK210BSMWAA6467LU/image.nii.gz"
    gt_path =  "./datasets/LA/nii_data/0RZDK210BSMWAA6467LU/image.nii.gz"
    category_index = 2  # 目标类别在gt标注中的索引
    output_dir = "./result/pred"
    roi_image, roi_label, meta_info = data_preprocess(img_path, gt_path, category_index=category_index)
    # 输出图像张量的大小与形状
    print(f"roi_image shape: {roi_image.shape}")
    print(f"roi_label shape: {roi_label.shape}")
    print(f"meta_info: {meta_info}")

    ''' 2. prepare the pre-trained model with local path or huggingface url '''
    # ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
    ckpt_path = "sam2_configs/sam_med3d_turbo.pth"
    # or you can use the local path like: ckpt_path = "./ckpt/sam_med3d_turbo.pth"
    model = medim.create_model("SAM-Med3D",
                               pretrained=True,
                               checkpoint_path=ckpt_path)
    
    ''' 3. infer with the pre-trained SAM-Med3D model '''
    roi_pred = sam_model_infer(model, roi_image, roi_gt=roi_label)
    print(f'roi_pred shape is {roi_pred.shape}')

    ''' 4. post-process and save the result '''
    output_path = osp.join(output_dir, osp.basename(img_path).replace(".nii.gz", "_pred.nii.gz"))
    data_postprocess(roi_pred, meta_info, output_path, img_path)

    print("result saved to", output_path)
