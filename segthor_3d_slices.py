#!/usr/bin/env python3

# You can run the following command to create a 3d dataset
# python segthor_3d_slices.py --source_dir ./data/segthor_train/train --dest_dir data/SEGTHOR_3D --train_size 30 --val_size 10 --process 4

import argparse
import pickle
import random
from pathlib import Path
from typing import Callable, Tuple, List, Dict
from multiprocessing import Pool

import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    return norm

def sanity_check(ct: np.ndarray, gt: np.ndarray) -> bool:
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8], gt.dtype
    assert set(np.unique(gt)) == set(range(5))
    return True

def process_patient(args: Tuple[str, Path, Path]) -> Tuple[str, Tuple[float, float, float]]:
    id_, dest_path, source_path = args
    patient_path: Path = source_path / id_

    ct_path: Path = patient_path / f"{id_}.nii.gz"
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    dx, dy, dz = nib_obj.header.get_zooms()

    gt_path: Path = patient_path / "GT.nii.gz"
    gt = np.asarray(nib.load(str(gt_path)).dataobj)
    assert sanity_check(ct, gt)

    norm_ct: np.ndarray = norm_arr(ct)

    combined_data = np.stack([norm_ct, gt], axis=-1)
    tensor_data = torch.from_numpy(combined_data)

    output_file = dest_path / f"{id_}.pt"
    torch.save(tensor_data, output_file)

    return id_, (dx, dy, dz)

def get_splits(src_path: Path, train_size: int, val_size: int) -> Tuple[List[str], List[str]]:
    ids: List[str] = sorted([p.name for p in src_path.glob('*') if p.is_dir()])
    print(f"Found {len(ids)} patients in the dataset")
    print(f"First 10 patient IDs: {ids[:10]}")
    
    total_samples = train_size + val_size
    assert len(ids) >= total_samples, f"Not enough samples in the dataset. Found {len(ids)}, need {total_samples}"

    random.shuffle(ids)
    train_ids = ids[:train_size]
    val_ids = ids[train_size:total_samples]

    assert len(train_ids) == train_size
    assert len(val_ids) == val_size

    return train_ids, val_ids

def check_paths(src_path: Path, dest_path: Path):
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_path}")
    if not src_path.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {src_path}")
    
    dest_path.mkdir(parents=True, exist_ok=True)

def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    try:
        check_paths(src_path, dest_path)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}")
        print("Please check the provided paths and ensure you have the correct permissions.")
        return

    train_ids, val_ids = get_splits(src_path, args.train_size, args.val_size)

    resolution_dict: Dict[str, Tuple[float, float, float]] = {}

    for mode, split_ids in zip(["train", "val"], [train_ids, val_ids]):
        dest_mode: Path = dest_path / mode
        dest_mode.mkdir(parents=True, exist_ok=True)
        print(f"Processing {len(split_ids)} patients to {dest_mode}")

        process_args = [(id_, dest_mode, src_path) for id_ in split_ids]
        
        if args.process == 1:
            results = [process_patient(arg) for arg in tqdm(process_args)]
        else:
            with Pool(args.process if args.process > 0 else None) as p:
                results = list(tqdm(p.imap(process_patient, process_args), total=len(split_ids)))

        resolution_dict.update(dict(results))

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionary to {f}")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='3D CT processing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--train_size', type=int, default=30, help="Number of patients for training set")
    parser.add_argument('--val_size', type=int, default=10, help="Number of patients for validation set")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--process', '-p', type=int, default=1, help="The number of cores to use for processing")
    args = parser.parse_args()
    random.seed(args.seed)
    print(args)
    return args

if __name__ == "__main__":
    main(get_args())