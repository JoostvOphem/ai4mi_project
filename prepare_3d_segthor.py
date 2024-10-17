import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable

import numpy as np
import nibabel as nib

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

def process_patient(id_: str, dest_path: Path, source_path: Path, test_mode: bool = False) -> tuple[float, float, float]:
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

    img_path: Path = dest_path / "img"
    gt_path: Path = dest_path / "gt"
    img_path.mkdir(parents=True, exist_ok=True)
    gt_path.mkdir(parents=True, exist_ok=True)

    nib.save(nib.Nifti1Image(norm_ct, nib_obj.affine, nib_obj.header), str(img_path / f"{id_}.nii.gz"))
    nib.save(nib.Nifti1Image(gt, gt_nib.affine, gt_nib.header), str(gt_path / f"{id_}.nii.gz"))

    return dx, dy, dz

def get_splits(src_path: Path, retains: int, fold: int) -> tuple[list[str], list[str], list[str]]:
    ids: list[str] = sorted(map_(lambda p: p.name, (src_path / 'train').glob('*')))
    print(f"Founds {len(ids)} in the id list")
    print(ids[:10])
    assert len(ids) > retains

    random.shuffle(ids)  # Shuffle before to avoid any problem if the patients are sorted in any way
    validation_slice = slice(fold * retains, (fold + 1) * retains)
    validation_ids: list[str] = ids[validation_slice]
    assert len(validation_ids) == retains

    training_ids: list[str] = [e for e in ids if e not in validation_ids]
    assert (len(training_ids) + len(validation_ids)) == len(ids)

    test_ids: list[str] = sorted(map_(lambda p: Path(p.stem).stem, (src_path / 'test').glob('*')))
    print(f"Founds {len(test_ids)} test ids")
    print(test_ids[:10])

    return training_ids, validation_ids, test_ids

def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the clean up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

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