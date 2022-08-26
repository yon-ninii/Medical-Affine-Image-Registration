import os
import numpy as np
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    RandAffined,
    Resized,
    ScaleIntensityRanged,
    EnsureTyped,
    Spacingd,
)
from monai.utils import set_determinism
import nibabel as nib


def train_val_to_df(data_dir) :
    '''
    Making files to dictionary type
    Files : pre_vein 1 ~ 113, pre_artery 1 ~ 113
    '''
    fix_vein_path = os.path.join(data_dir, 'pre_vein')
    mov_artery_path = os.path.join(data_dir, 'pre_artery')

    train_dicts = [
        {
            "fixed_image": os.path.join(fix_vein_path,
                                        "pre_vein%d.nii.gz" % idx),
            "moving_image": os.path.join(mov_artery_path,
                                        "pre_artery%d.nii.gz" % idx),
        }
        for idx in range(1, 90)
    ]
    
    val_dicts = [
        {
            "fixed_image": os.path.join(fix_vein_path,
                                        "pre_vein%d.nii.gz" % idx),
            "moving_image": os.path.join(mov_artery_path,
                                        "pre_artery%d.nii.gz" % idx),
        }
        for idx in range(90, 113)
    ]

    train_files = train_dicts
    val_files = val_dicts

    return train_files, val_files

def test_to_df(data_dir) :

    fix_vein_path = os.path.join(data_dir, 'pre_vein')
    mov_artery_path = os.path.join(data_dir, 'pre_artery')
    
    test_dicts = [
        {
            "fixed_image": os.path.join(fix_vein_path,
                                        "pre_vein%d.nii.gz" % idx),
            "moving_image": os.path.join(mov_artery_path,
                                        "pre_artery%d.nii.gz" % idx),
        }
        for idx in range(1, 2)
    ]

    test_files = test_dicts

    return test_files



def set_transforms(mode = 'train') :

    set_determinism(seed=42)

    if mode == 'train': 
        train_transforms = Compose(
            [
                LoadImaged(
                    keys=["fixed_image", "moving_image"]
                ),
                AddChanneld(
                    keys=["fixed_image", "moving_image"]
                ),
                ScaleIntensityRanged(
                    keys=["fixed_image", "moving_image"],
                    a_min=-150, a_max=300, b_min=0.0, b_max=1.0, clip=True,
                ),
                Resized(
                    keys=["fixed_image", "moving_image"],
                    mode=('trilinear', 'trilinear'), # interpolation
                    align_corners=(True, True),
                    spatial_size=(64, 64, 32)
                ),
                EnsureTyped(
                    keys=["fixed_image", "moving_image"]
                ),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(
                    keys=["fixed_image", "moving_image"]
                ),
                AddChanneld(
                    keys=["fixed_image", "moving_image"]
                ),
                ScaleIntensityRanged(
                    keys=["fixed_image", "moving_image"],
                    a_min=-150, a_max=300, b_min=0.0, b_max=1.0, clip=True,
                ),
                Resized(
                    keys=["fixed_image", "moving_image"],
                    mode=('trilinear', 'trilinear'), # interpolation
                    align_corners=(True, True),
                    spatial_size=(64, 64, 32)
                ),
                EnsureTyped(
                    keys=["fixed_image", "moving_image"]
                ),
            ]
        )


        return train_transforms, val_transforms

    
    elif mode == 'test':
        test_transforms = Compose(
            [
                LoadImaged(
                    keys=["fixed_image", "moving_image"]
                ),
                AddChanneld(
                    keys=["fixed_image", "moving_image"]
                ),
                ScaleIntensityRanged(
                    keys=["fixed_image", "moving_image"],
                    a_min=-150, a_max=300, b_min=0.0, b_max=1.0, clip=True,
                ),
                Resized(
                    keys=["fixed_image", "moving_image"],
                    mode=('trilinear', 'trilinear'), # interpolation
                    align_corners=(True, True),
                    spatial_size=(84, 84, 42)
                ),
                EnsureTyped(
                    keys=["fixed_image", "moving_image"]
                ),
            ]
        )
        
        origin_transforms = Compose(
            [
                LoadImaged(
                    keys=["fixed_image", "moving_image"]
                ),
                AddChanneld(
                    keys=["fixed_image", "moving_image"]
                ),
                ScaleIntensityRanged(
                    keys=["fixed_image", "moving_image"],
                    a_min=-150, a_max=300, b_min=0.0, b_max=1.0, clip=True,
                ),
                Resized(
                    keys=["fixed_image", "moving_image"],
                    mode=('trilinear', 'trilinear'), # interpolation
                    align_corners=(True, True),
                    spatial_size=(512, 512, 200)
                ),
                EnsureTyped(
                    keys=["fixed_image", "moving_image"]
                ),
            ]
        )

        return test_transforms, origin_transforms
