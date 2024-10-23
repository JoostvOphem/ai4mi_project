from pathlib import Path
import os
import shutil

datasets = ["SEGTHOR_3D", "SEGTHOR_3D_MED", "SEGTHOR_3D_AI", "SEGTHOR_3D_AI"]
path_raw = Path("nnUNet_stored") / "raw"
path_stored = Path("nnUNet_stored") / "stored"
path_preprocessed = Path("nnUNet_stored") / "preprocessed"

for i, dataset_name in enumerate(datasets):

    dataset_number = "{:03d}".format(i+1)


    path_to_data_1 = Path("..") / "data" / dataset_name / "train"
    path_to_data_2 = Path("..") / "data" / dataset_name / "val"
    paths = [path_to_data_1, path_to_data_2]

    dataset_folder_path_raw = path_raw / f"Dataset{dataset_number}_{dataset_name}"
    imagesTr_folder_path_raw = dataset_folder_path_raw / "imagesTr"
    labelsTr_folder_path_raw = dataset_folder_path_raw / "labelsTr"



    GTs = []
    Patients = []
    for path in paths:

        for folder in os.listdir(path):
            if folder == "gt":
                for name in os.listdir(path / folder):
                    GTs.append(path / folder / name)
            
            elif folder == "img":
                for name in os.listdir(path / folder):
                    Patients.append(path / folder / name)

    for j, GT in enumerate(GTs):
        num = "{:03d}".format(j+1)
        shutil.copy(GT, labelsTr_folder_path_raw / f"1a_{num}.nii.gz")

    for j, Patient in enumerate(Patients):
        num = "{:03d}".format(j+1)
        shutil.copy(Patient, imagesTr_folder_path_raw / f"1a_{num}_0000.nii.gz")
