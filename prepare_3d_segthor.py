import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable, Literal

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

from utils import map_, tqdm_

def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    return norm

def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()
    assert 0.896 <= dx <= 1.37, dx
    assert dx == dy
    assert 2 <= dz <= 3.7, dz
    assert (x, y) == (512, 512)
    assert 135 <= z <= 284, z
    return True

def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8], gt.dtype
    assert set(np.unique(gt)) == set(range(5))
    return True

def apply_interpolation(img: np.ndarray, gt: np.ndarray, target_shape=(128, 128, 64)) -> tuple[np.ndarray, np.ndarray]:
    """Apply spline interpolation to resize the volume."""
    zoom_factors = [t/s for t, s in zip(target_shape, img.shape)]
    
    # Apply interpolation
    img_resized = zoom(img, zoom_factors, order=3, mode='nearest')  # cubic spline for image
    gt_resized = zoom(gt, zoom_factors, order=0, mode='nearest')    # nearest neighbor for labels
    
    return img_resized, gt_resized

def apply_sliding_window(img: np.ndarray, gt: np.ndarray, target_shape=(128, 128, 64)) -> tuple[np.ndarray, np.ndarray]:
    """Extract a central patch of the target size."""
    D, H, W = img.shape
    tD, tH, tW = target_shape
    
    # Calculate start indices for central crop
    start_d = max(0, (D - tD) // 2)
    start_h = max(0, (H - tH) // 2)
    start_w = max(0, (W - tW) // 2)
    
    # Extract patches
    img_patch = img[start_d:start_d+tD, start_h:start_h+tH, start_w:start_w+tW].copy()
    gt_patch = gt[start_d:start_d+tD, start_h:start_h+tH, start_w:start_w+tW].copy()
    
    # Pad if necessary
    if img_patch.shape != target_shape:
        pad_d = max(0, tD - img_patch.shape[0])
        pad_h = max(0, tH - img_patch.shape[1])
        pad_w = max(0, tW - img_patch.shape[2])
        
        img_patch = np.pad(img_patch, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        gt_patch = np.pad(gt_patch, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
    
    return img_patch, gt_patch

def process_patient(id_: str, dest_path: Path, source_path: Path, 
                   preprocessing: Literal["interpolation", "window"],
                   test_mode: bool = False) -> tuple[float, float, float]:
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    ct_path: Path = (id_path / f"{id_}.nii.gz") if not test_mode else (source_path / "test" / f"{id_}.nii.gz")
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    dx, dy, dz = nib_obj.header.get_zooms()

    assert sanity_ct(ct, *ct.shape, *nib_obj.header.get_zooms())

    if not test_mode:
        gt_path: Path = id_path / "GT.nii.gz"
        gt_nib = nib.load(str(gt_path))
        gt = np.asarray(gt_nib.dataobj)
        assert sanity_gt(gt, ct)
    else:
        gt = np.zeros_like(ct, dtype=np.uint8)

    norm_ct: np.ndarray = norm_arr(ct)

    # Apply preprocessing
    if preprocessing == "interpolation":
        norm_ct, gt = apply_interpolation(norm_ct, gt)
    else:  # window
        norm_ct, gt = apply_sliding_window(norm_ct, gt)

    img_path: Path = dest_path / "img"
    gt_path: Path = dest_path / "gt"
    img_path.mkdir(parents=True, exist_ok=True)
    gt_path.mkdir(parents=True, exist_ok=True)

    nib.save(nib.Nifti1Image(norm_ct, nib_obj.affine, nib_obj.header), str(img_path / f"{id_}.nii.gz"))
    nib.save(nib.Nifti1Image(gt, gt_nib.affine, gt_nib.header), str(gt_path / f"{id_}.nii.gz"))

    return dx, dy, dz

def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    base_dest_path: Path = Path(args.dest_dir)

    assert src_path.exists()
    
    # Create both dataset versions
    for preprocessing in ["interpolation", "window"]:
        dest_path = base_dest_path / f"SEGTHOR_3D_{'128' if preprocessing == 'interpolation' else 'Window'}"
        if dest_path.exists():
            print(f"Destination {dest_path} already exists, skipping...")
            continue
            
        print(f"\nProcessing dataset with {preprocessing} method to {dest_path}")
        
        training_ids: list[str]
        validation_ids: list[str]
        test_ids: list[str]
        training_ids, validation_ids, test_ids = get_splits(src_path, args.retains, args.fold)

        resolution_dict: dict[str, tuple[float, float, float]] = {}

        split_ids: list[str]
        for mode, split_ids in zip(["train", "val", "test"], [training_ids, validation_ids, test_ids]):
            dest_mode: Path = dest_path / mode
            print(f"Processing {len(split_ids)} pairs to {dest_mode}")

            pfun: Callable = partial(process_patient,
                                   dest_path=dest_mode,
                                   source_path=src_path,
                                   preprocessing=preprocessing,
                                   test_mode=mode == 'test')
            resolutions: list[tuple[float, float, float]]
            iterator = tqdm_(split_ids)
            match args.process:
                case 1:
                    resolutions = list(map(pfun, iterator))
                case -1:
                    resolutions = Pool().map(pfun, iterator)
                case _ as p:
                    resolutions = Pool(p).map(pfun, iterator)

            for key, val in zip(split_ids, resolutions):
                resolution_dict[key] = val

        with open(dest_path / "spacing.pkl", 'wb') as f:
            pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
            print(f"Saved spacing dictionary to {f}")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='3D data preparation parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--retains', type=int, default=10, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--process', '-p', type=int, default=1, help="The number of cores to use for processing")
    args = parser.parse_args()
    random.seed(args.seed)
    print(args)
    return args

if __name__ == "__main__":
    main(get_args())