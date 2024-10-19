import skimage.morphology as mrph
from load_segmentation import Results, GT
from pathlib import Path

if __name__ == "__main__":
    r = Results(Path("results") / "segthor_unet" / "ce")
    gts = GT(Path("data") / "SEGTHOR" / "val" / "gt")
    for array, patid in r.best_epoch_val_3d():
        print(patid)
        gt = gts.patient_volume(patid)
