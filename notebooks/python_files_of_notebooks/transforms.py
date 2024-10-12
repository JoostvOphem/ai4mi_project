import numpy as np
from scipy.ndimage import affine_transform

import matplotlib.pyplot as plt
import copy
from batchgenerators.transforms.utility_transforms import OneOfTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.color_transforms import (
    BrightnessTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    Rot90Transform,
    TransposeAxesTransform,
)
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.noise_transforms import MedianFilterTransform, GaussianBlurTransform
from batchgenerators.transforms.noise_transforms import SharpeningTransform
from batchgenerators.transforms.abstract_transforms import Compose

import copy
import random
import scipy.ndimage as ndi
from scipy.ndimage import uniform_filter, gaussian_filter
from monai.transforms import Transform
from skimage import img_as_float, img_as_uint
from skimage.util import random_noise
from batchgenerators.transforms.abstract_transforms import AbstractTransform

def apply_and_visualize_transform(transform, image_data, slice_number, title, GT=None):
    # Create a deep copy of the image data to prevent in-place modifications
    image_data_copy = copy.deepcopy(image_data)
    # If no transform is provided, use the original image data
    if transform is None:
        augmented_data = {'data': image_data_copy}
    else:
        # Apply the transform
        augmented_data = transform(data=image_data_copy)
    # Extract the augmented image
    augmented_image = augmented_data['data'][0, 0]
    # Visualize the augmented image slice with proper scaling
    plt.imshow(
        augmented_image[:, :, slice_number],
        cmap='gray')
    plt.title(title)
    plt.colorbar(label='HU Value')
    plt.show()

    if GT is not None:
        apply_and_visualize_transform(transform, GT, slice_number, "GT")
    return

class PartialVolumeArtifactTransform(AbstractTransform):
    def __init__(self, kernel_size_range=(3, 5)):
        # Define a range for the kernel size
        self.kernel_size_range = kernel_size_range

    def __call__(self, **data_dict):
        data = data_dict['data']
        
        # Randomly select a kernel size from the given range
        kernel_size = random.randint(*self.kernel_size_range)

        # Apply uniform filtering using the random kernel size
        data = uniform_filter(data, size=kernel_size, mode='reflect')
        data_dict['data'] = data
        return data_dict
    
class StreakArtifactTransform(AbstractTransform):
    def __init__(self, num_streaks=10, intensity_decrease=500):
        self.num_streaks = num_streaks
        self.intensity_decrease = intensity_decrease

    def __call__(self, **data_dict):
        data = data_dict['data']
        for _ in range(self.num_streaks):
            x = np.random.randint(0, data.shape[2])
            data[:, :, x, :, :] -= self.intensity_decrease
        data_dict['data'] = data
        return data_dict
    
class RingArtifactTransform(Transform):
    def __init__(self, frequency_range=(2, 50), amplitude_range=(10, 500)):
        self.frequency = np.random.uniform(*frequency_range)
        self.amplitude = np.random.uniform(*amplitude_range)

    def __call__(self, **data_dict):
        data = data_dict['data']
        rows, cols = data.shape[2], data.shape[3]
        center = (rows / 2, cols / 2)
        y, x = np.indices((rows, cols))
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        ring_pattern = self.amplitude * np.sin(2 * np.pi * r / self.frequency)
        # Expand ring_pattern to match the data shape: (1,1,x,y,z)
        ring_pattern_expanded = ring_pattern[np.newaxis, np.newaxis, :, :, np.newaxis]
        # Add ring pattern across all slices
        data += ring_pattern_expanded
        # Clip to maintain HU range
        data = np.clip(data, -1000, 3000)
        data_dict['data'] = data
        return data_dict
        
class ZebraArtifactTransform(AbstractTransform):
    def __init__(self, stripe_frequency_range=(5, 15), amplitude_range=(30, 70)):
        # Instead of fixed values, we now define ranges for stripe frequency and amplitude
        self.stripe_frequency_range = stripe_frequency_range
        self.amplitude_range = amplitude_range

    def __call__(self, **data_dict):
        data = data_dict['data']

        # Randomly choose a stripe frequency and amplitude from the given ranges
        stripe_frequency = random.uniform(*self.stripe_frequency_range)
        amplitude = random.uniform(*self.amplitude_range)

        X = np.arange(data.shape[2])

        # Create the zebra pattern using the randomly chosen stripe frequency and amplitude
        zebra_pattern = (np.sin(2 * np.pi * X / stripe_frequency) > 0).astype(float) * amplitude - (amplitude / 2)
        zebra_pattern = zebra_pattern[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        
        # Add the zebra pattern to the data
        data_dict['data'] = data + zebra_pattern
        return data_dict

def spatial_transform(patch_size, percent_scaling, percent_rotation):
    return SpatialTransform(
        patch_size=patch_size,
        patch_center_dist_from_border=None,
        do_elastic_deform=False,
        do_rotation=True,
        angle_x=(-30 / 360 * 2. * np.pi, 30 / 360 * 2. * np.pi),
        angle_y=(-30 / 360 * 2. * np.pi, 30 / 360 * 2. * np.pi),
        angle_z=(-30 / 360 * 2. * np.pi, 30 / 360 * 2. * np.pi),
        p_rot_per_axis=1.0,  # Ensure rotation is applied
        do_scale=True,
        scale=(0.7, 1.43),
        border_mode_data="constant",
        border_cval_data=0,
        order_data=3,
        border_mode_seg="constant",
        border_cval_seg=-1,
        order_seg=1,
        random_crop=False,
        p_el_per_sample=0.0,  # No elastic deformation
        p_scale_per_sample=percent_scaling,  # Ensure scaling is applied
        p_rot_per_sample=percent_rotation,  # Ensure rotation is applied
        independent_scale_for_each_axis=True,
    )

def gaussian_noise_transform(mu, sigma, p=1): 
    return GaussianNoiseTransform(
        noise_variance=(mu, sigma), #0, 0.1 orignially
        p_per_sample=p  # chance noise will be applied
    )

def gaussian_blur_transform(blur_sigma, p=1):
    return GaussianBlurTransform(
        blur_sigma=blur_sigma,
        different_sigma_per_channel=True,
        p_per_sample=p,
        p_per_channel=1.0
    )

def brightness_transform(mu, sigma, p=1):
    return BrightnessTransform(
        mu=mu,
        sigma=sigma, # 0.5 (normal amount for range 0-1) * relevant range (from -1000, 1000)
        per_channel=True,
        p_per_sample=p,
        p_per_channel=1.0
    )

def contrast_transform(range, p=1): 
    return ContrastAugmentationTransform(
        contrast_range=range,
        preserve_range=True,
        per_channel=True,
        data_key='data',
        p_per_sample=p,
        p_per_channel=1.0
    )

def lowres_transform(zoom_range, p=1):
    return SimulateLowResolutionTransform(
        zoom_range=zoom_range,
        per_channel=True,
        p_per_channel=1.0,
        order_downsample=0,
        order_upsample=3,
        p_per_sample=p,
        ignore_axes=None
    )

def gamma_transform(ranges, p=1):
    return GammaTransform(
        gamma_range=ranges,
        invert_image=True,
        per_channel=True,
        retain_stats=True,
        p_per_sample=p
    )

def mirror_transform():
    return MirrorTransform(
        axes=(0, 1, 2)
    )

# def black_rectangle_transform(patch_size):
#     rectangle_size = [[max(1, p // 10), p // 3] for p in patch_size]

#     # BlankRectangleTransform parameters
#     blank_rectangle_transform = BlankRectangleTransform(
#         rectangle_size=rectangle_size,
#         rectangle_value=np.mean,
#         num_rectangles=(1, 5),
#         force_square=False,
#         p_per_sample=1.0,
#         p_per_channel=1.0
#     )

def sharpening_transform(strength, p=1):
    return SharpeningTransform(
        strength=strength,
        same_for_each_channel=False,
        p_per_sample=p,
        p_per_channel=1.0
    )

def rot_90_transform(num_rot, p=1):
    return Rot90Transform(
        num_rot=num_rot,
        p_per_sample=p
    )

def transpose_axes_transform(transpose_any_of_these, p=1):
    return TransposeAxesTransform(
        transpose_any_of_these=transpose_any_of_these,
        p_per_sample=p
    )