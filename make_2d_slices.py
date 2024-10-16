import os
import argparse
import nibabel as nib
import numpy as np
from PIL import Image
import tqdm
from skimage.transform import resize

def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm
    assert 0 == res.min(), res.min()
    assert res.max() == 255, res.max()
    return res.astype(np.uint8)

def create_2d_slices(input_path, output_path, file_prefix, data_type):
    img = nib.load(input_path)
    img_data = img.get_fdata()
    
    # Normalize the entire 3D image
    if data_type == 'gt':
        # For ground truth, we assume it's already binary (0 and 1)
        normalized_data = (img_data * 255).astype(np.uint8)
    else:
        # For regular images, use the provided normalization function
        normalized_data = norm_arr(img_data)
    
    num_slices = normalized_data.shape[2]
    
    for i in range(num_slices):
        slice_2d = normalized_data[:, :, i]
        
        # Resize to 256x256
        resized_slice = resize(slice_2d, (256, 256), order=1, preserve_range=True).astype(np.uint8)
        
        # Convert to PIL Image
        img_slice = Image.fromarray(resized_slice, mode='L')  # 'L' mode for 8-bit grayscale
        
        # Save the image
        img_slice.save(os.path.join(output_path, f"{file_prefix}_{i:04d}.png"))

def process_dataset(input_root, output_root):
    for subset in ['train', 'val']:
        for data_type in ['img', 'gt']:
            input_path = os.path.join(input_root, subset, data_type)
            output_path = os.path.join(output_root, subset, data_type)
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            # Process each file in the input directory
            for filename in tqdm.tqdm(os.listdir(input_path), desc=f"Processing {subset} {data_type}"):
                if filename.endswith('.nii.gz') or filename.endswith('.nii'):
                    patient_id = filename.split('.')[0]
                    input_file = os.path.join(input_path, filename)
                    create_2d_slices(input_file, output_path, patient_id, data_type)

def main():
    parser = argparse.ArgumentParser(description="Process 3D medical images to 2D slices.")
    parser.add_argument("--src", required=True, help="Source folder containing the SEGTHOR_3D dataset")
    parser.add_argument("--dest", required=True, help="Destination folder for the processed 2D slices")
    args = parser.parse_args()

    input_root = args.src
    output_root = args.dest

    process_dataset(input_root, output_root)
    print(f"Processing complete. 2D slices have been saved in the {output_root} directory.")

if __name__ == "__main__":
    main()