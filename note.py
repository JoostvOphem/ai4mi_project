import os
import nibabel as nib
from pathlib import Path

def print_nifti_info(file_path):
    nii_img = nib.load(file_path)
    header = nii_img.header
    
    print(f"File: {file_path}")
    print(f"  Shape: {nii_img.shape}")
    print(f"  Data type: {nii_img.get_data_dtype()}")
    print(f"  Voxel dimensions: {header.get_zooms()[:3]}")
    print(f"  Unit: {header.get_xyzt_units()[0]}")
    print("  Value range:", nii_img.get_fdata().min(), "to", nii_img.get_fdata().max())
    print()

def process_directory(base_path):
    base_path = Path(base_path)
    
    # Process CT images
    ct_dir = base_path / "train" / "img"
    print("CT Images:")
    for file in sorted(ct_dir.glob("*.nii.gz")):
        print_nifti_info(file)
    
    # Process Ground Truth images
    gt_dir = base_path / "train" / "gt"
    print("Ground Truth Images:")
    for file in sorted(gt_dir.glob("*.nii.gz")):
        print_nifti_info(file)

if __name__ == "__main__":
    data_path = "data/SEGTHOR_3D"
    process_directory(data_path)