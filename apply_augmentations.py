import nibabel as nib
import matplotlib.pyplot as plt

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
    """Save the middle slice of the 3D data as a PNG file."""
    middle_slice = data[:, :, data.shape[2]//2]
    plt.imshow(middle_slice, cmap='gray')
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    # File paths
    ct_input_file = "data/SEGTHOR_3D/train/img/Patient_03.nii.gz"
    gt_input_file = "data/SEGTHOR_3D/train/gt/Patient_03.nii.gz"

    
    # PNG output files
    ct_original_png = "ct_original_middle_slice.png"
    ct_augmented_png = "ct_augmented_middle_slice.png"
    gt_original_png = "gt_original_middle_slice.png"
    gt_augmented_png = "gt_augmented_middle_slice.png"

    # Load the CT and GT NIfTI files
    ct_data, _ = load_nifti(ct_input_file)
    gt_data, _ = load_nifti(gt_input_file)

    # Save middle slices of original CT and GT as PNG
    save_middle_slice_png(ct_data, ct_original_png)
    save_middle_slice_png(gt_data, gt_original_png)
    print(f"Original CT middle slice saved to {ct_original_png}")
    print(f"Original GT middle slice saved to {gt_original_png}")

    # Apply augmentation to both CT and GT
    # augmented_ct, augmented_gt = lower_resolution(ct_data, gt_data)
    augmented_ct = apply_zebra_artifact(ct_data)

    # Save middle slices of augmented CT and GT as PNG
    save_middle_slice_png(augmented_ct, ct_augmented_png)
    # save_middle_slice_png(augmented_gt, gt_augmented_png)
    print(f"Augmented CT middle slice saved to {ct_augmented_png}")
    print(f"Augmented GT middle slice saved to {gt_augmented_png}")



if __name__ == "__main__":
    main()