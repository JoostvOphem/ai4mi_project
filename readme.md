# AI for medical imaging — Fall 2024 course project

Alessandro, Erik, Joost, Rory & Taiki

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

### Getting the data
Generation the data, via the recipe in the `Makefile`:
```
$ make data/TOY2
$ make data/SEGTHOR
```

The synthetic dataset is generated randomly, whereas for Segthor it is required to put the file [`segthor_train.zip`](https://amsuni-my.sharepoint.com/:u:/g/personal/h_t_g_kervadec_uva_nl/EfMdFte7pExAnPwt4tYUcxcBbJJO8dqxJP9r-5pm9M_ARw?e=ZNdjee) (required a UvA account) in the `data/` folder. If the computer running it is powerful enough, the recipe for `data/SEGTHOR` can be modified in the [Makefile](Makefile) to enable multi-processing (`-p -1` option, see `python slice_segthor.py --help` or its code directly).

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