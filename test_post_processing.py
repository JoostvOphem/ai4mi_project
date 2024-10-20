from utils import dice_batch
from src.modules.load_segmentation import Results, GT, to_onehot_tensor, save_as_nii
from pathlib import Path
from torch import from_numpy
from src.modules.post_processing import closing, opening, closing_opening
import glob
import json
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

test_root = Path("test_pp")

def create_processed_volumes():
    test_root.mkdir(parents=True, exist_ok=True)
    r = Results(Path("results") / "segthor_unet" / "ce")
    # gts = GT(Path("data") / "SEGTHOR" / "val" / "gt")
    for pred, patid in r.best_epoch_val_3d():
        patient_str = "Patient_{:02}".format(patid)
        # save_as_nii(pred, test_root / f"{patient_str}_pred")
        # for i in range(10):
        #     save_as_nii(closing(pred, i), test_root / f"{patient_str}_closing_{i}")
        for i in range(5, 7):
            save_as_nii(opening(pred, i), test_root / f"{patient_str}_opening_{i}")
        # save_as_nii(closing(pred, 0), test_root / f"{patient_str}_closing_0")
        # save_as_nii(closing_opening(pred, 4, 2), test_root / f"{patient_str}_c_4_o_2")
        # save_as_nii(closing_opening(pred, 6, 4), test_root / f"{patient_str}_c_6_o_4")
        # save_as_nii(closing_opening(pred, 8, 6), test_root / f"{patient_str}_c_8_o_6")
        # gt = gts.patient_volume(patid)
        # print(dice_batch(pred, gt))


def compare_dice():
    gts = GT(Path("data") / "SEGTHOR" / "val" / "gt")
    json_path = test_root / "scores.json"
    if json_path.exists():
        with open(json_path, "r") as file:
            scores = json.load(file)
    else:
        scores = {}
    for patid in [1,2,13,16,21,22,28,30,35,39]:
        patient_str = "Patient_{:02}".format(patid)
        print(patient_str)
        gt = gts.patient_volume(patid)
        for path in sorted(glob.glob((test_root / f"{patient_str}_*.nii*").as_posix())):
            name = Path(path).stem.replace(patient_str + "_", "")
            pred = to_onehot_tensor(np.asarray(nib.load(str(path)).dataobj))
            dice = dice_batch(pred, gt)
            print(dice)
            scores.setdefault(name, []).append(dice.tolist())
            with open(test_root / "scores.json", "w") as file:
                json.dump(scores, file, indent=2)


def plot_dice():
    json_path = test_root / "scores.json"
    with open(json_path, "r") as file:
        scores = json.load(file)
    baselines = np.array(scores['pred']).mean(axis=0)[1:]
    baselines = np.concatenate((np.ravel(baselines), np.array([baselines.mean(axis=0)])))

    x = range(-6, 10)
    dices = []
    for rad in x:
        key = f"closing_{rad}" if rad >= 0 else f"opening_{-rad}"
        dices.append(np.array(scores[key]).mean(axis=0)[1:])
    dices = np.array(dices).T
    dices = np.vstack((dices, dices.mean(axis=0)))

    fig = plt.figure(figsize=(10,8))
    for i, (label, color) in enumerate(zip(['Esophagus', 'Heart', 'Trachea', 'Aorta', 'mean'], ['#80ae80', '#f1d691', '#b17a65', '#6fb8d2', 'black'])):
        plt.axhline(baselines[i], linestyle='--', color=color, linewidth=1)
        plt.plot(x, dices[i], label=label, color=color, linewidth=2)
    
    plt.legend()
    plt.xlabel('Closing radius in voxels (negative is opening)')
    plt.ylabel('Average 3D Dice score')
    plt.title('Average Dice score after post-processing with opening and LCC')

    plt.show()

    # plt.tight_layout()
    # plt.savefig(test_root / "post_process_dice.png")


if __name__ == "__main__":
    # create_processed_volumes()
    # compare_dice()
    plot_dice()