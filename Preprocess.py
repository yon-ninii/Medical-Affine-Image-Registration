import os
import numpy as np
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    Resized,
    EnsureTyped,
    SpatialCrop,
    KeepLargestConnectedComponent
)
from monai.utils import set_determinism
from monai.data import NibabelReader, write_nifti
from monai.data import DataLoader, CacheDataset, Dataset, IterableDataset
import nibabel as nib
from tqdm import tqdm


def pre_to_df(data_dir='/home/ym_regi/home/ppoohh/CT_data') :

    fix_vein_path = os.path.join(data_dir, 'nii_vein')
    mov_artery_path = os.path.join(data_dir, 'nii_artery')
    mask_artery_path = os.path.join(data_dir, 'label_artery')
    mask_vein_path = os.path.join(data_dir, 'label_vein')

    idxs = []
    
    for idx in range(1, 130):
        f_name = "%04d.nii.gz" % idx
        if os.path.exists(os.path.join(fix_vein_path, f_name)) and os.path.exists(os.path.join(mov_artery_path, f_name)):
            idxs.append(idx)

    train_dicts = [
        {
            "fixed_image": os.path.join(fix_vein_path,
                                        "%04d.nii.gz" % idx),
            "moving_image": os.path.join(mov_artery_path,
                                        "%04d.nii.gz" % idx),
            "mask_artery_image": os.path.join(mask_artery_path,
                                        "%04d" % idx, 'Liver.nii.gz'),
            "mask_vein_image": os.path.join(mask_vein_path,
                                        "%04d" % idx, 'Liver.nii.gz'),
        }
        for idx in idxs
    ]
    
    train_dicts += [
        {
            "fixed_image": os.path.join(fix_vein_path,
                                        "S%02d.nii.gz" % idx),
            "moving_image": os.path.join(mov_artery_path,
                                        "S%02d.nii.gz" % idx),
            "mask_artery_image": os.path.join(mask_artery_path,
                                        "S%02d" % idx, 'Liver.nii.gz'),
            "mask_vein_image": os.path.join(mask_vein_path,
                                        "S%02d" % idx, 'Liver.nii.gz'),
        }
        for idx in range(1, 37)
    ]

    train_files = train_dicts

    return train_files
    
def set_transforms() :

    set_determinism(seed=0)

    pre_transforms = Compose(
        [
            LoadImaged(
                keys=["fixed_image", "moving_image", "mask_artery_image", "mask_vein_image"]
            ),
            AddChanneld(
                keys=["fixed_image", "moving_image", "mask_artery_image", "mask_vein_image"]
            ),
            EnsureTyped(
                keys=["fixed_image", "moving_image", "mask_artery_image", "mask_vein_image"]
            ),
        ]
    )
    return pre_transforms

    
def mid_z(mask_image):
    max_z, min_z = 0, 500
    x, y, z = mask_image.shape
    for k in range(z - 1, 0, -1):
        if np.sum(mask_image[:, :, k]) > 100:
            max_z = k
            break
    return max_z #(max_z + min_z) // 2

pre_file = pre_to_df()

seg_trans = KeepLargestConnectedComponent(applied_labels=[1], is_onehot=True)

for i in range(len(pre_file)):
    moving_image, fixed_image = pre_file[i]['moving_image'], pre_file[i]['fixed_image']
    mask_mov, mask_fix = pre_file[i]['mask_artery_image'], pre_file[i]['mask_vein_image']
    moving, fixed = nib.load(moving_image), nib.load(fixed_image)
    masked_mov, masked_fix = nib.load(mask_mov), nib.load(mask_fix)
    move = moving.get_fdata()
    fix = fixed.get_fdata()
    mask_mov = seg_trans(masked_mov.get_fdata())
    mask_fix = seg_trans(masked_fix.get_fdata())
    move_affine = moving.affine
    fix_affine = fixed.affine
    mid1 = mid_z(mask_mov)
    mid2 = mid_z(mask_fix)
    print(i, 'move:', move.shape)
    print(i, 'fix:', fix.shape)
    #cropper1 = SpatialCrop(roi_center = (256, mid1, 256), roi_size = (512, 200, 512))
    #cropper2 = SpatialCrop(roi_center = (256, mid2, 256), roi_size = (512, 200, 512))
    cropper1 = SpatialCrop(roi_start = (0, mid1 - 200, 0), roi_end = (512, mid1, 512))
    cropper2 = SpatialCrop(roi_start = (0, mid2 - 200, 0), roi_end = (512, mid2, 512))
    pre_mov = cropper1(move)
    pre_fix = cropper2(fix)
    pre_mov = nib.Nifti1Image(pre_mov, affine = move_affine)
    pre_fix = nib.Nifti1Image(pre_fix, affine = fix_affine)
    print('pre_move:', pre_mov.shape)
    print('pre_fix:', pre_fix.shape)
    print('mid1:', mid1, 'mid2:', mid2)
    nib.save(pre_mov, f'/home/ym_regi/home/ppoohh/CT_data/pre_artery/pre_artery{i + 1}.nii.gz')
    nib.save(pre_fix, f'/home/ym_regi/home/ppoohh/CT_data/pre_vein/pre_vein{i + 1}.nii.gz')

print('finished preprocessing')