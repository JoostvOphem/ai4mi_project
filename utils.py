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
from functools import partial
from multiprocessing import Pool
from contextlib import AbstractContextManager
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import Tensor, einsum
import nibabel as nib
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries as skimage_find_boundaries

tqdm_ = partial(tqdm, dynamic_ncols=True,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]')


class Dcm(AbstractContextManager):
    # Dummy Context manager
    def __exit__(self, *args, **kwargs):
        pass


# Functools
A = TypeVar("A")
B = TypeVar("B")


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def starmmap_(fn: Callable[[Tuple[A]], B], iter: Iterable[Tuple[A]]) -> List[B]:
    return Pool().starmap(fn, iter)


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res


def probs2class(probs: Tensor) -> Tensor:
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


# Save the raw predictions
def save_images(segs: Tensor, names: Iterable[str], root: Path, is_3d: bool = False) -> None:
    for seg, name in zip(segs, names):
        save_path = (root / name).with_suffix(".png" if not is_3d else ".nii.gz")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        seg_np = seg.detach().cpu().numpy()
        
        if not is_3d:
            # For 2D data
            Image.fromarray(seg_np.astype(np.uint8)).save(save_path)
        else:
            # For 3D data
            if len(seg_np.shape) == 3:
                # If it's already 3D, save as NIfTI
                # Convert to uint8 or int16 depending on the range of values
                if seg_np.max() <= 255:
                    seg_np = seg_np.astype(np.uint8)
                else:
                    seg_np = seg_np.astype(np.int16)
                nib.save(nib.Nifti1Image(seg_np, np.eye(4)), str(save_path))
            elif len(seg_np.shape) == 4:
                # If it's 4D (e.g., with a channel dimension), save middle slice as PNG
                middle_slice = seg_np.shape[0] // 2
                Image.fromarray(seg_np[middle_slice].astype(np.uint8)).save(save_path.with_suffix(".png"))
            else:
                raise ValueError(f"Unexpected shape for 3D data: {seg_np.shape}")


# Metrics
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")  # used for 3d dice

def meta_precision(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    pred_size: Tensor = einsum(sum_str, [pred]).type(torch.float32)

    precision: Tensor = (inter_size + smooth) / (pred_size + smooth)
    return precision


precision_coef = partial(meta_precision, "bk...->bk")
precision_batch = partial(meta_precision, "bk...->k")  # used for 3d precision

# Recall
def meta_recall(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    label_size: Tensor = einsum(sum_str, [label]).type(torch.float32)

    recall: Tensor = (inter_size + smooth) / (label_size + smooth)
    return recall


recall_coef = partial(meta_recall, "bk...->bk")
recall_batch = partial(meta_recall, "bk...->k")  # used for 3d recall

# Jaccard Index (IoU)
def meta_jaccard(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    union_size: Tensor = einsum(sum_str, [union(label, pred)]).type(torch.float32)

    jaccard: Tensor = (inter_size + smooth) / (union_size + smooth)
    return jaccard


jaccard_coef = partial(meta_jaccard, "bk...->bk")
jaccard_batch = partial(meta_jaccard, "bk...->k")  # used for 3d jaccard (IoU)


def metric_coef(pred_seg, gt, metric):
	if metric == 'dice':
		return dice_coef(pred_seg, gt)
	
	elif metric == 'jaccard':
		return jaccard_coef(pred_seg, gt)
		
	elif metric == 'precision':
		return precision_coef(pred_seg, gt)
	
	elif metric == 'recall':
		return recall_coef(pred_seg, gt)
	
	elif metric == 'nsd':
		return nsd(pred_seg, gt, threshold=1.0)
		
	elif metric == 'masd':
		return masd(pred_seg, gt)
		
	else:
		raise ValueError(f"Unsupported metric: {metric}")

def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a & b
    assert sset(res, [0, 1])

    return res


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a | b
    assert sset(res, [0, 1])

    return res


def downsample_boundary(boundary, factor=10):
    return boundary[::factor]

def masd(pred_seg, gt_seg):
    # Find the boundaries of the predicted and ground truth segmentations
    pred_boundary = find_boundaries(pred_seg)
    gt_boundary = find_boundaries(gt_seg)

    d_pred_to_gt = compute_average_surface_distance(pred_boundary, gt_boundary)
    d_gt_to_pred = compute_average_surface_distance(gt_boundary, pred_boundary)
    return (d_pred_to_gt + d_gt_to_pred) / 2


def nsd(pred_seg, gt_seg, threshold=1.0, downsample_factor=10):
    pred_boundary = find_boundaries(pred_seg)
    gt_boundary = find_boundaries(gt_seg)

    pred_boundary = downsample_boundary(pred_boundary, downsample_factor)
    gt_boundary = downsample_boundary(gt_boundary, downsample_factor)
    
    d_pred_to_gt = compute_average_surface_distance(pred_boundary, gt_boundary)
    d_gt_to_pred = compute_average_surface_distance(gt_boundary, pred_boundary)

    # Compute NSD based on a given threshold
    nsd_value = np.mean(d_pred_to_gt < threshold) * np.mean(d_gt_to_pred < threshold)
    
    return nsd_value

def compute_average_surface_distance(boundary1, boundary2):
    boundary1_coords = np.argwhere(boundary1)
    boundary2_coords = np.argwhere(boundary2)

    if boundary1_coords.size == 0 or boundary2_coords.size == 0:
        return np.inf  # Return infinity if one of the boundaries is empty

    # Create KDTree for boundary points
    tree1 = cKDTree(boundary1_coords)
    tree2 = cKDTree(boundary2_coords)
    distances1, _ = tree1.query(boundary2_coords)
    distances2, _ = tree2.query(boundary1_coords)

    avg_distance1 = distances1.mean() if distances1.size > 0 else np.inf
    avg_distance2 = distances2.mean() if distances2.size > 0 else np.inf
    return (avg_distance1 + avg_distance2) / 2
    
    

def find_boundaries(segmentation, mode='outer'):
    # If the input is a PyTorch tensor, move it to the CPU and convert to NumPy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    
    segmentation = np.asarray(segmentation)

    # 2D
    if segmentation.ndim == 2:
        return skimage_find_boundaries(segmentation, mode=mode)

    # 3D
    elif segmentation.ndim == 3:
        boundaries = np.zeros_like(segmentation, dtype=np.bool_)
        for i in range(segmentation.shape[0]):  # For each slice
            boundaries[i] = skimage_find_boundaries(segmentation[i], mode=mode)
        return boundaries

    # 4D
    elif segmentation.ndim == 4:
        boundaries = np.zeros_like(segmentation, dtype=np.bool_)
        for b in range(segmentation.shape[0]):  # For each batch
            for i in range(segmentation.shape[1]):  # For each slice in the batch
                boundaries[b, i] = skimage_find_boundaries(segmentation[b, i], mode=mode)
        return boundaries

    else:
        raise ValueError("Segmentation map must be 2D or 3D.")