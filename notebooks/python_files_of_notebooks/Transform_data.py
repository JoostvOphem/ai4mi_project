import numpy as np
import nibabel as nib
from pathlib import Path

# --------------------------------------

real_27 = nib.load(Path("..") / "data" / "segthor_train" / "train" / "Patient_27" / "GT2.nii.gz")
fake_27 = nib.load(Path("..") / "data" / "segthor_train" / "train" / "Patient_27" / "GT.nii.gz")
real_27_array = np.array(real_27.dataobj)
fake_27_array = np.array(fake_27.dataobj)

# look only to transform the heart
real_27_array = (real_27_array == 2)
fake_27_array = (fake_27_array == 2)


# find translation vector
def find_centroid(array):
    indices = np.argwhere(array == 1)  # Find points where array == 1
    centroid = np.mean(indices, axis=0)  # Compute mean of these points
    return centroid

centroid_real = find_centroid(real_27_array)
centroid_fake = find_centroid(fake_27_array)

translation_vector = centroid_real - centroid_fake

# ----------------------------------------

# tranlate the heart of the specific patient

def translate_array(array, translation_vector):
    # Find the indices of the non-zero (1) elements
    indices = np.argwhere(array == 1)
    
    # Apply translation to the indices
    translated_indices = indices + translation_vector
    
    # Create a new array with the same shape as the input array
    translated_array = np.zeros_like(array)
    
    # Set the translated indices to 1
    for ind in translated_indices.astype(int):
        translated_array[tuple(ind)] = 1
    
    return translated_array

# ---------------------------------------

def save_array_ass_nii(array, filename, fake_GT):
    """
    Converts a 3D NumPy array into a NIfTI (.nii) file.
    
    Args:
        array (np.ndarray): 3D NumPy array to be converted.
        filename (str): Path to save the NIfTI file (with .nii extension).
    
    Returns:
        None
    """
    # Convert NumPy array to a NIfTI image
    nii_image = nib.Nifti1Image(array, affine=fake_GT.affine, header=fake_GT.header)
    
    # Save the NIfTI image to a file
    nib.save(nii_image, filename)

# ----------------------------------------

# iterate over patients
for patient_number in range(1, 41):
    if patient_number < 10:
        patient_number = "0" + str(patient_number)
    fake_nii = nib.load(Path("..") / "data" / "segthor_train" / "train" / f"Patient_{patient_number}" / "GT.nii.gz")
    fake_array = np.array(fake_nii.dataobj)

    # save non-heart organs
    saved_fake_array = np.copy(fake_array)
    saved_fake_array[saved_fake_array == 2] = 0

    # mask with only the heart
    fake_array = (fake_array == 2)

    translated_fake_array = translate_array(fake_array, translation_vector)

    # re-add the other organs
    saved_fake_array[translated_fake_array == 1] = 2

    # save the found array
    save_array_ass_nii(saved_fake_array, Path("..") / "data" / "segthor_train" / "train" / f"Patient_{patient_number}" / "real_GT.nii.gz", fake_nii)

