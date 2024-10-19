from pathlib import Path
import re
import numpy as np
import glob
from PIL import Image

_2d_pattern = re.compile(r"""Patient_(?P<patid>\d+)_(?P<layer>\d+)""", re.VERBOSE)

def stack_2d_imgs(path: Path, patient_id: int):
    mask = (path / "Patient_{:02}_*.png".format(patient_id)).as_posix()
    imgs = sorted(glob.glob(mask))
    arrays = [np.array(Image.open(img)) for img in imgs]
    return np.dstack(arrays)

def parse_2d_filepath(path: Path):
    match = _2d_pattern.match(path.stem)
    if match is None:
        return None
    patid = int(match.group("patid"))
    layer = int(match.group("layer"))
    return {"patient_id":patid, "layer":layer}

class Results:
    def __init__(self, path: Path):
        assert path.is_dir()
        self._root = path

        if (self._root / "best_epoch" / "val").is_dir():
            self._best_epoch_val_patients = set()
            for pth in (self._root / "best_epoch" / "val").iterdir():
                metadata = parse_2d_filepath(pth)
                if metadata is not None:
                    self._best_epoch_val_patients.add(metadata["patient_id"])
    
    
    # returns 3d NumPy arrays of the predictions for the patients in the validation set
    def best_epoch_val_3d(self):
        assert self._best_epoch_val_patients is not None
        for patid in sorted(list(self._best_epoch_val_patients)):
            yield stack_2d_imgs(self._root / "best_epoch" / "val", patid), patid


class GT:
    def __init__(self, path: Path):
        assert path.is_dir()
        self._root = path

        self._patients = set()
        for pth in self._root.iterdir():
            metadata = parse_2d_filepath(pth)
            if metadata is not None:
                self._patients.add(metadata["patient_id"])
    
    def patient_volume(self, patient_id: int):
        assert patient_id in self._patients
        return stack_2d_imgs(self._root, patient_id)
    
    

