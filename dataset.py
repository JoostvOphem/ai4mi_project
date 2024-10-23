#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from typing import Callable, Union, Tuple, List

import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def make_dataset(root, subset) -> list[tuple[Path, Path]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images = sorted(img_path.glob("*.png"))
    full_labels = sorted(full_path.glob("*.png"))

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor = self.img_transform(Image.open(img_path))
        gt: Tensor = self.gt_transform(Image.open(gt_path))

        _, W, H = img.shape
        K, _, _ = gt.shape
        assert gt.shape == (K, W, H)

        return {"images": img,
                "gts": gt,
                "stems": img_path.stem}


class VolumeDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None, gt_transform=None, debug=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.files = make_volume_dataset(root_dir, subset)
        
        if debug:
            self.files = self.files[:10]
        print(f">> Created {subset} volume dataset with {len(self)} volumes...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[torch.Tensor, int, str]]:
        img_path, gt_path = self.files[index]
        
        # Load nifti files
        img_nifti = nib.load(str(img_path))
        gt_nifti = nib.load(str(gt_path))
        
        # Get data as numpy arrays
        img_np = img_nifti.get_fdata().astype(np.float32)
        gt_np = gt_nifti.get_fdata().astype(np.float32)
        
        # Add channel dimension if not present
        if len(img_np.shape) == 3:
            img_np = img_np[np.newaxis, ...]  # Add channel dimension
        
        # Apply transformations
        if self.img_transform:
            img = self.img_transform(img_np)
        else:
            img = torch.from_numpy(img_np)
        
        if self.gt_transform:
            gt = self.gt_transform(gt_np)
        else:
            gt = torch.from_numpy(gt_np.astype(np.int64))
        
        # Ensure correct dimensions
        assert img.dim() == 4, f"Expected 4D tensor for image, got shape {img.shape}"
        assert gt.dim() == 4, f"Expected 4D tensor for ground truth, got shape {gt.shape}"
        
        return {
            "images": img,
            "gts": gt,
            "stems": img_path.stem
        }