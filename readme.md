# AI for medical imaging — Fall 2024 course project

Alessandro, Erik, Joost, Rory & Taiki

## Setting up the environment
Run the following commands to create a needed environment:
```
$ git clone https://github.com/JoostvOphem/ai4mi_project.git 
$ cd ai4mi_project
$ python -m venv ai4mi
$ source ai4mi/bin/activate
$ python -m pip install -r requirements.txt
$ python -V # Make sure that your Python version is >=3.11
```

### Getting the data
Generation the data, via the recipe in the `Makefile`:
```
# This creates 2D slices of SEGTHOR dataset in the shape [256, 256]
$ make data/SEGTHOR

# Make a 3D folder in the original shape ([512, 512, z]) for preprocessing purpose
$ python prepare_3d_segthor.py --source_dir data/segthor_train --dest_dir data/SEGTHOR_3D --retains 10
```

The synthetic dataset is generated randomly, whereas for Segthor it is required to put the file [`segthor_train.zip`](https://amsuni-my.sharepoint.com/:u:/g/personal/h_t_g_kervadec_uva_nl/EfMdFte7pExAnPwt4tYUcxcBbJJO8dqxJP9r-5pm9M_ARw?e=ZNdjee) (required a UvA account) in the `data/` folder. If the computer running it is powerful enough, the recipe for `data/SEGTHOR` can be modified in the [Makefile](Makefile) to enable multi-processing (`-p -1` option, see `python slice_segthor.py --help` or its code directly).

### Preprocessing
First fix the heart placements running
$ python notebooks/python_files_of_notebooks/official_transform_data.py

This makes use of the rotation and translation matrices given for the use of this course. 
An alternate approach which looks similarly good for the patients but works differently can be found in notebooks/python_files_of_notebooks/Transform_data.py. We decided againt using this transformation because we deemed it less likely that there were mistakes with the official solution.
Finally, an approach that also explicitly finds the rotation and translation matrices can be found on notebooks/Transform_data_solution_on_all.ipynb

After this run the following commands to make our preprocessed data folders: SEGTHOR_3D_MED, SEGTHOR_3D_AI, SEGTHOR_3D_ALL, SEGTHOR_MED, SEGTHOR_AI, SEGTHOR_ALL
```
$ python apply_augmentations.py --mode MED
$ python apply_augmentations.py --mode AI
$ python apply_augmentations.py --mode ALL
$ python make_2d_slices.py --src data/SEGTHOR_3D_MED --dest data/SEGTHOR_MED
$ python make_2d_slices.py --src data/SEGTHOR_3D_AI --dest data/SEGTHOR_AI
$ python make_2d_slices.py --src data/SEGTHOR_3D_ALL --dest data/SEGTHOR_ALL

# Or simply run this job file
$ sbatch job_files/create_preprocessed_data.job
```

## Running on Snellius
After creating the environment you need to run the following lines to run the main.py

```
$ source ai4mi/bin/activate
$ module load 2022
$ module load Python/3.10.4-GCCcore-11.3.0

# Ask for the GPU node can be done by running the line below or make a job file and run with sbatch command.
$ srun --partition=gpu --gpus=1 --ntasks=1 --cpus-per-task=18 --time=00:59:00 --pty bash -i

# You can run the following command to train and get result of a model.
$ python -O main.py --model enet --dataset SEGTHOR --mode full --epoch 50 --dest results/segthor_enet/ce --gpu --metric dice
```


## Project overview
The project is based around the SegTHOR challenge data, which was kindly allowed by Caroline Petitjean (challenge organizer) to use for the course. The challenge was originally on the segmentation of different organs: heart, aorta, esophagus and trachea.
![Segthor Overview](segthor_overview.png)


### How to replicate our results
To replicate our results you must run the following command lines given that you already have the SEGTHOR datatset and the correct environment.
```
$ sbatch apply_data_augmentation.job
$ sbatch make_2d_slices_from_3d_folders.job
```

The following job files contain instructions to train models on each preprocessed dataset.
```
root folder/
│
├── job_files/
│   ├── no_augmentation.job
│   ├── med_augmentation.job
│   ├── ai_augmentation.job
│   └── all_augmentation.job
```
