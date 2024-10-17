from scipy import ndimage
import numpy as np
from scipy.ndimage import zoom


def sharpen_3d(image, alpha=0.1):
    """Apply 3D sharpening filter to the image."""
    blurred = ndimage.gaussian_filter(image, sigma=1)
    return image + alpha * (image - blurred)

def spatial_3d(image, gt):
    """Rotate 3D image and ground truth along z-axis while preserving voxel shape."""
    angle_degrees = np.random.uniform(-3, 3)
    angle_radians = np.deg2rad(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])
    center = np.array(image.shape) // 2
    offset = center - np.dot(rotation_matrix, center)
    
    rotated_image = ndimage.affine_transform(image, rotation_matrix.T, offset=offset, order=1, mode='constant', cval=0)
    rotated_gt = ndimage.affine_transform(gt, rotation_matrix.T, offset=offset, order=0, mode='constant', cval=0)
    
    return rotated_image, rotated_gt

def mirror_3d(image, gt):
    """Apply horizontal mirroring to both image and ground truth."""
    return np.flip(image, axis=1), np.flip(gt, axis=1)

def gamma_correction(image, gamma_range=(0.8, 1.2)):
    """Apply random gamma correction to the image."""
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    min_val = image.min()
    max_val = image.max()
    
    # Normalize to [0, 1], apply gamma, then scale back
    normalized = (image - min_val) / (max_val - min_val)
    corrected = np.power(normalized, gamma)
    return corrected * (max_val - min_val) + min_val

def gaussian_blur_3d(image, sigma_range=(0.1, 1.0)):
    """Apply random Gaussian blur to the image."""
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    return ndimage.gaussian_filter(image, sigma=sigma)

def rotate_90_or_270(image, gt):
    """Apply random 90 or 270-degree rotation to both image and ground truth."""
    k = np.random.choice([1, 3])  # 1: 90°, 3: 270°
    return np.rot90(image, k=k, axes=(0, 1)), np.rot90(gt, k=k, axes=(0, 1))

def add_gaussian_noise(image, mean=0, std_range=(0.003, 0.005)):
    """
    Add Gaussian noise to the image.
    """
    std = np.random.uniform(std_range[0], std_range[1])
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    
    # Clip the values to maintain the original range
    return np.clip(noisy_image, image.min(), image.max())

def lower_resolution(ct_image, gt_image, scale_range=(0.7, 0.9)):
    """
    Lower the resolution of the CT image while carefully handling the GT for organ segmentation.
    """
    # Randomly choose a scale factor
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    
    # Lower resolution of CT image
    downscaled_ct = zoom(ct_image, (scale_factor, scale_factor, scale_factor), order=1)
    upscaled_ct = zoom(downscaled_ct, (1/scale_factor, 1/scale_factor, 1/scale_factor), order=1)
    lowres_ct = zoom(upscaled_ct, np.array(ct_image.shape) / np.array(upscaled_ct.shape), order=1)
    
    # Handle GT image
    # Downsample GT using nearest neighbor to preserve label integrity
    downscaled_gt = zoom(gt_image, (scale_factor, scale_factor, scale_factor), order=0)
    
    # Upsample back to original size using nearest neighbor
    upscaled_gt = zoom(downscaled_gt, (1/scale_factor, 1/scale_factor, 1/scale_factor), order=0)
    
    # Ensure the GT has the same shape as the input
    adjusted_gt = zoom(upscaled_gt, np.array(gt_image.shape) / np.array(upscaled_gt.shape), order=0)
    
    return lowres_ct, adjusted_gt

def adjust_brightness(ct_image, brightness_range=(0.7, 0.9)):
    """
    Adjust the brightness of the CT image.
    """
    # Randomly choose a brightness factor
    brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
    # Apply brightness adjustment
    adjusted_image = ct_image * brightness_factor
    
    # Clip values to maintain the original range
    return np.clip(adjusted_image, ct_image.min(), ct_image.max())

def adjust_contrast_ct(ct_image, contrast_range=(1.01, 1.03)):
    """
    Adjust the contrast of the CT image while maintaining the original range.
    """
    # Store original min and max
    original_min = ct_image.min()
    original_max = ct_image.max()

    # Randomly choose a contrast factor
    contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
    
    # Compute the middle of the original range
    midpoint = (original_max + original_min) / 2

    # Apply contrast adjustment
    adjusted_image = (ct_image - midpoint) * contrast_factor + midpoint
    
    # Clip to original range
    adjusted_image = np.clip(adjusted_image, original_min, original_max)

    return adjusted_image

def apply_streak_artifact(ct_image, num_streaks=10, intensity_decrease=0.4):
    """
    Apply streak artifacts to a CT image.
    """
    for _ in range(num_streaks):
        x = np.random.randint(0, ct_image.shape[0])
        ct_image[x, :, :] *= intensity_decrease
    return ct_image


def apply_ring_artifact(ct_image):
    """
    Apply realistic cylindrical ring artifacts to a CT image.
    """
    frequency = np.random.uniform(1, 5)
    amplitude = np.random.uniform(0.001, 0.01)

    rows, cols = ct_image.shape[0], ct_image.shape[1]
    min, max = ct_image.min(), ct_image.max()
    center = (rows / 2, cols / 2)
    y, x = np.indices((rows, cols))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    ring_pattern = amplitude * np.sin(2 * np.pi * r / frequency)

    for z in range(ct_image.shape[2]):
        ct_image[:, :, z] += ring_pattern

    # Clip to maintain HU range
    ct_image = np.clip(ct_image, min, max)

    return ct_image

def apply_zebra_artifact(ct_image, stripe_frequency_range=(1, 5), amplitude_range=(0.001, 0.01)):
    """
    Apply zebra artifacts to a CT image and optionally to ground truth.
    """
    min, max = ct_image.min(), ct_image.max()
    stripe_frequency = np.random.uniform(*stripe_frequency_range)
    amplitude = np.random.uniform(*amplitude_range)
    
    X = np.arange(ct_image.shape[0])
    zebra_pattern = (np.sin(2 * np.pi * X / stripe_frequency) > 0).astype(float) * amplitude - (amplitude / 2)
    
    ct_image += zebra_pattern[:, np.newaxis, np.newaxis]
    ct_image = np.clip(ct_image, min, max)
    return ct_image