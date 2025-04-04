import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Paths for training and validation datasets
TRAIN_DATASET_PATH = '../input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = '../input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'

# Define a custom Dataset for loading the BraTS data
class BraTSDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load images and mask
        flair_image = nib.load(self.image_paths[idx][0]).get_fdata()
        t1_image = nib.load(self.image_paths[idx][1]).get_fdata()
        t1ce_image = nib.load(self.image_paths[idx][2]).get_fdata()
        t2_image = nib.load(self.image_paths[idx][3]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        # Display dimensions of each file
        print(f"Dimensions of flair image: {flair_image.shape}")
        print(f"Dimensions of t1 image: {t1_image.shape}")
        print(f"Dimensions of t1ce image: {t1ce_image.shape}")
        print(f"Dimensions of t2 image: {t2_image.shape}")
        print(f"Dimensions of mask: {mask.shape}")

        # Convert images and mask to torch tensors
        flair_image = torch.tensor(flair_image, dtype=torch.float32)
        t1_image = torch.tensor(t1_image, dtype=torch.float32)
        t1ce_image = torch.tensor(t1ce_image, dtype=torch.float32)
        t2_image = torch.tensor(t2_image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        sample = {
            'flair': flair_image,
            't1': t1_image,
            't1ce': t1ce_image,
            't2': t2_image,
            'mask': mask
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

# Paths for each image modality and corresponding mask
image_paths = []
mask_paths = []

# Example for one subject (adjust as needed for multiple subjects)
subject_path = os.path.join(TRAIN_DATASET_PATH, 'BraTS20_Training_001')
image_paths.append([os.path.join(subject_path, 'BraTS20_Training_001_flair.nii'),
                    os.path.join(subject_path, 'BraTS20_Training_001_t1.nii'),
                    os.path.join(subject_path, 'BraTS20_Training_001_t1ce.nii'),
                    os.path.join(subject_path, 'BraTS20_Training_001_t2.nii')])

mask_paths.append(os.path.join(subject_path, 'BraTS20_Training_001_seg.nii'))

# Create the dataset
dataset = BraTSDataset(image_paths=image_paths, mask_paths=mask_paths)

# Create DataLoader for batching
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Fetch a sample from the dataset and display images
sample = dataset[0]

# Extract images and mask
test_image_flair = sample['flair'].numpy()
test_image_t1 = sample['t1'].numpy()
test_image_t1ce = sample['t1ce'].numpy()
test_image_t2 = sample['t2'].numpy()
test_mask = sample['mask'].numpy()

# Plotting the central slices for each image modality and mask
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))
slice_w = 25
ax1.imshow(test_image_flair[:, :, test_image_flair.shape[0]//2-slice_w], cmap='gray')
ax1.set_title('Image flair')
ax2.imshow(test_image_t1[:, :, test_image_t1.shape[0]//2-slice_w], cmap='gray')
ax2.set_title('Image t1')
ax3.imshow(test_image_t1ce[:, :, test_image_t1ce.shape[0]//2-slice_w], cmap='gray')
ax3.set_title('Image t1ce')
ax4.imshow(test_image_t2[:, :, test_image_t2.shape[0]//2-slice_w], cmap='gray')
ax4.set_title('Image t2')
ax5.imshow(test_mask[:, :, test_mask.shape[0]//2-slice_w])
ax5.set_title('Mask')
plt.show()