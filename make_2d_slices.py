import os
import argparse
import nibabel as nib
import numpy as np
from PIL import Image
import tqdm

def create_2d_slices(input_path, output_path, file_prefix):
    img = nib.load(input_path)
    img_data = img.get_fdata()
    num_slices = img_data.shape[2]

    for i in range(num_slices):
        slice_2d = img_data[:, :, i]
        slice_2d = np.clip(slice_2d * 255, 0, 255).astype(np.uint8)
        img_slice = Image.fromarray(slice_2d, mode='L')  # 'L' mode for 8-bit grayscale

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
            for filename in tqdm.tqdm(os.listdir(input_path)):
                if filename.endswith('.nii.gz'):
                    patient_id = filename.split('.')[0]
                    input_file = os.path.join(input_path, filename)
                    create_2d_slices(input_file, output_path, patient_id)

def main():
    parser = argparse.ArgumentParser(description="Process 3D medical images to 2D slices.")
    parser.add_argument("--source", required=True, help="Source folder containing the SEGTHOR_3D dataset")
    parser.add_argument("--destination", required=True, help="Destination folder for the processed 2D slices")
    args = parser.parse_args()

    input_root = args.source
    output_root = args.destination

    process_dataset(input_root, output_root)
    print(f"Processing complete. 2D slices have been saved in the {output_root} directory.")

if __name__ == "__main__":
    main()