from utils import dice_batch
from src.modules.load_segmentation import Results, GT
from pathlib import Path
from torch import from_numpy

if __name__ == "__main__":
    r = Results(Path("results") / "segthor_unet" / "ce")
    gts = GT(Path("data") / "SEGTHOR" / "val" / "gt")
    for pred, patid in r.best_epoch_val_3d():
        print(pred.shape)
        gt = gts.patient_volume(patid)
        print(dice_batch(pred, gt))