import os
import argparse
import shutil
import nibabel as nib
from itertools import cycle
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from augmentation_functions import (sharpen_3d,
                                    spatial_3d,
                                    mirror_3d,
                                    gamma_correction,
                                    gaussian_blur_3d,
                                    rotate_90_or_270,
                                    add_gaussian_noise,
                                    lower_resolution,
                                    adjust_brightness,
                                    adjust_contrast_ct,
                                    apply_streak_artifact,
                                    apply_ring_artifact,
                                    apply_zebra_artifact
                                    )

def load_nifti(file_path):
    """Load NIfTI file and return image data and affine."""
    nifti = nib.load(file_path)
    return nifti.get_fdata(), nifti.affine

def save_nifti(file_path, data, affine):
    """Save data as NIfTI file."""
    nifti = nib.Nifti1Image(data, affine)
    nib.save(nifti, file_path)

def save_middle_slice_png(data, file_path):
    """Save the middle slice of the 3D data as a PNG file. Handy to see the result on data preprocessing.
    Example Usage:
    ct_input_file = "data/SEGTHOR_3D/train/img/Patient_03.nii.gz"
    gt_input_file = "data/SEGTHOR_3D/train/gt/Patient_03.nii.gz"

    # Load the CT and GT NIfTI files
    ct_data, _ = load_nifti(ct_input_file)
    gt_data, _ = load_nifti(gt_input_file)

    save_middle_slice_png(ct_data, "ct_original_middle_slice.png")
    save_middle_slice_png(gt_data, "gt_original_middle_slice.png")

    # Apply augmentation
    augmented_ct, augmented_gt = lower_resolution(ct_data, gt_data)   # for both Ct & GT
    # augmented_ct = apply_zebra_artifact(ct_data)    # for CT only

    save_middle_slice_png(augmented_ct, ct_augmented_png)
    save_middle_slice_png(augmented_gt, gt_augmented_png)
    """
    middle_slice = data[:, :, data.shape[2]//2] # shows (x,y)-plane in z.max//2
    plt.imshow(middle_slice, cmap='gray')
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def create_transform_iterators(mode):
    """Create cycling iterators for each transform category."""
    major_funcs = cycle([add_gaussian_noise, lower_resolution, adjust_brightness, adjust_contrast_ct])
    
    if mode == 'MED':
        medium_funcs = cycle([gamma_correction, gaussian_blur_3d, sharpen_3d])
        minor_funcs = cycle([apply_streak_artifact, apply_ring_artifact, apply_zebra_artifact])
    elif mode == 'AI':
        medium_funcs = cycle([gamma_correction, gaussian_blur_3d, spatial_3d])
        minor_funcs = cycle([sharpen_3d, mirror_3d, rotate_90_or_270])
    else:  # ALL mode
        medium_funcs = cycle([gamma_correction, gaussian_blur_3d, sharpen_3d, spatial_3d])
        minor_funcs = cycle([apply_streak_artifact, apply_ring_artifact, apply_zebra_artifact,
                             sharpen_3d, mirror_3d, rotate_90_or_270])
    
    return major_funcs, medium_funcs, minor_funcs

def apply_transform(ct_data, gt_data, transform_func):
    """Apply the given transform function to the data."""
    if transform_func in [lower_resolution, rotate_90_or_270, mirror_3d, spatial_3d]:
        return transform_func(ct_data, gt_data)
    else:
        return transform_func(ct_data), gt_data

def main(args):
    dest = f"data/SEGTHOR_3D_{args.mode}"
    
    # Create destination folders if they don't exist
    os.makedirs(os.path.join(dest, 'train', 'img'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'train', 'gt'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'val', 'img'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'val', 'gt'), exist_ok=True)

    # Process training data
    train_img_folder = os.path.join(args.src, 'train', 'img')
    train_gt_folder = os.path.join(args.src, 'train', 'gt')

    # Copy original files
    for filename in os.listdir(train_img_folder):
        if filename.endswith('.nii.gz'):
            shutil.copy(os.path.join(train_img_folder, filename), os.path.join(dest, 'train', 'img', filename))
            shutil.copy(os.path.join(train_gt_folder, filename), os.path.join(dest, 'train', 'gt', filename))

    # Get the number of original files
    original_count = len([f for f in os.listdir(train_img_folder) if f.endswith('.nii.gz')])
    
    # Calculate the number of new files to create
    new_files_count = original_count * (args.size - 1)
    
    # Calculate the number of files for each transform category
    major_count = int(new_files_count * 0.45)
    medium_count = int(new_files_count * 0.35)
    minor_count = new_files_count - major_count - medium_count  # Ensure we use all remaining files

    # Create cycling iterators for transform functions
    major_funcs, medium_funcs, minor_funcs = create_transform_iterators(args.mode)

    print(f"Starting augmentation process. Creating {new_files_count} new files.")

    # Create new augmented files
    for i in tqdm(range(new_files_count), desc="Augmenting files", unit="file"):
        # Randomly select an original file
        original_file = random.choice(os.listdir(train_img_folder))
        
        # Load CT scan and ground truth
        ct_path = os.path.join(train_img_folder, original_file)
        gt_path = os.path.join(train_gt_folder, original_file)
        ct_data, ct_affine = load_nifti(ct_path)
        gt_data, gt_affine = load_nifti(gt_path)
        
        # Apply transformation based on the current count
        if i < major_count:
            transform_func = next(major_funcs)
        elif i < major_count + medium_count:
            transform_func = next(medium_funcs)
        else:
            transform_func = next(minor_funcs)
        
        augmented_ct, augmented_gt = apply_transform(ct_data, gt_data, transform_func)
        
        # Save augmented data with a new filename
        new_filename = f"augmented_{i+1}_{original_file}"
        save_nifti(os.path.join(dest, 'train', 'img', new_filename), augmented_ct, ct_affine)
        save_nifti(os.path.join(dest, 'train', 'gt', new_filename), augmented_gt, gt_affine)

    # Copy validation data without augmentation
    valid_img_folder = os.path.join(args.src, 'val', 'img')
    valid_gt_folder = os.path.join(args.src, 'val', 'gt')

    for filename in os.listdir(valid_img_folder):
        if filename.endswith('.nii.gz'):
            shutil.copy(os.path.join(valid_img_folder, filename), os.path.join(dest, 'val', 'img', filename))
            shutil.copy(os.path.join(valid_gt_folder, filename), os.path.join(dest, 'val', 'gt', filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data augmentation for SEGTHOR_3D dataset")
    parser.add_argument("--src", type=str, default="data/SEGTHOR_3D", help="Source directory containing the original dataset")
    parser.add_argument("--mode", type=str, default="MED", choices=["MED", "AI", "ALL"], help="Augmentation mode")
    parser.add_argument("--size", type=int, default=2, help="Size multiplier for the augmented dataset")
    args = parser.parse_args()

    if args.size < 2:
        raise ValueError("Size must be at least 2 to create additional data")

    main(args)