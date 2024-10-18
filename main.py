#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import SliceDataset, VolumeDataset
from ENet import ENet
from Models import UNet, UNETR_monai, shallowCNN
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   jaccard_coef,
                   precision_coef,
                   recall_coef,
                   nsd,
                   masd,
                   metric_coef,
                   save_images)

from losses import (CrossEntropy,
                    DiceLoss,
                    CombinedLoss,
                    FocalLoss)


datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the clases with C (often used for the number of Channel)
models = {'enet':ENet, 'unet':UNet, 'unetr':UNETR_monai, 'shallowcnn':shallowCNN}
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8}
datasets_params["SEGTHOR_MED"] = {'K': 5, 'net': UNet, 'B': 8}
datasets_params["SEGTHOR_AI"] = {'K': 5, 'net': ENet, 'B': 8} 
datasets_params["SEGTHOR_ALL"] = {'K': 5, 'net': UNet, 'B': 8} 
datasets_params["SEGTHOR_3D"] = {'K': 5, 'net': UNETR_monai, 'B': 4, 'img_shape': (128, 128, 64), 'input_dim': 1}

losses = {'ce':  CrossEntropy, 'dice': DiceLoss, 'focal': FocalLoss, 'combine':CombinedLoss}

# edited from: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int, str]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    net = models[args.model]

    if args.dataset == "SEGTHOR_3D":
        img_shape = datasets_params[args.dataset]['img_shape']
        input_dim = datasets_params[args.dataset]['input_dim']
        net = net(input_dim=input_dim, output_dim=K)
    else:
        net = net(1, K)
    
    if hasattr(net, 'init_weights'):
        net.init_weights()

    net.to(device)

    lr = 0.0005
    if args.optim == 'adamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    root_dir = Path("data") / args.dataset

    if args.dataset == "SEGTHOR_3D":
        img_transform = transforms.Compose([
            lambda nd: nd / nd.max(),  # Normalize to [0, 1]
            lambda nd: torch.tensor(nd, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        ])

        gt_transform = transforms.Compose([
            lambda nd: torch.tensor(nd, dtype=torch.int64),
            lambda t: class2one_hot(t.unsqueeze(0), K=K),
            lambda t: t.squeeze(0)  # Remove the extra dimension added by class2one_hot
        ])
    else:
        # Keep the existing 2D transforms
        img_transform = transforms.Compose([
            lambda img: img.convert('L'),
            lambda img: np.array(img)[np.newaxis, ...],
            lambda nd: nd / 255,  # max <= 1
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

        gt_transform = transforms.Compose([
            lambda img: np.array(img)[...],
            lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
            lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
            lambda t: class2one_hot(t, K=K),
            itemgetter(0)
        ])
    
    if args.dataset == "SEGTHOR_3D":
        train_set = VolumeDataset('train',
                            root_dir,
                            img_transform=img_transform,
                            gt_transform=gt_transform,
                            debug=args.debug)
        val_set = VolumeDataset('val',
                        root_dir,
                        img_transform=img_transform,
                        gt_transform=gt_transform,
                        debug=args.debug)
    else:
        train_set = SliceDataset('train',
                                root_dir,
                                img_transform=img_transform,
                                gt_transform=gt_transform,
                                debug=args.debug)
        val_set = SliceDataset('val',
                            root_dir,
                            img_transform=img_transform,
                            gt_transform=gt_transform,
                            debug=args.debug)
    
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=5,
                              shuffle=True)

    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=5,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K, args.metric)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    (net, optimizer, device, train_loader, val_loader, K, metric) = setup(args)

    if args.mode == "full" and args.model != "unetr":
        loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset in ['SEGTHOR', 'SEGTHOR_3D', 'SEGTHOR_STUDENTS']:
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    elif args.mode == "full" and args.model == "unetr":
        loss_fn = DiceLoss(idk=list(range(K)))
    else:
        raise ValueError(args.mode, args.dataset)
    
    early_stopper = EarlyStopper(patience=args.early_stopping_patience, min_delta=args.early_stopping_min_delta)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0
    best_epoch: int = -1

    for e in range(args.epochs):
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    
                    # Handle both 2D and 3D data
                    if args.dataset == "SEGTHOR_3D":
                        B, C, D, H, W = img.shape
                    else:
                        B, C, H, W = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = metric_coef(pred_seg, gt, metric)  # One Metric value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            # Instead of saving images here, we'll store them temporarily
                            temp_save_dir = args.dest / f"temp_iter{e:03d}" / m
                            save_images(predicted_class,
                                        data['stems'],
                                        temp_save_dir,
                                        is_3d=(args.dataset == "SEGTHOR_3D"))
                        early_stopper(loss.item())

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(0, K)}
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            print(f">>> Improved Metric at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC")
            best_dice = current_dice
            best_epoch = e

            # Save the best epoch information
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(str(e))

            # Move the temporary saved images to the best_epoch folder
            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"temp_iter{e:03d}", best_folder)

            # Save the model and weights
            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")

        # Remove the temporary saved images for this epoch
        rmtree(args.dest / f"temp_iter{e:03d}")

        if early_stopper.early_stop:
            print(f">>> Stopping early at epoch {e}.")
            break

    # After all epochs, ensure only the best epoch's images are kept
    for e in range(args.epochs):
        if e != best_epoch:
            temp_dir = args.dest / f"temp_iter{e:03d}"
            if temp_dir.exists():
                rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='enet', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=list(datasets_params.keys()))
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")
    parser.add_argument('--metric', default='dice', choices=['dice', 'jaccard', 'precision', 'recall', 'nsd', 'masd'],  # Add metric argument
                        help="Metric to evaluate the model (default: dice).")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logic around epochs and logging easily.")
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help="Amount of non-improving epochs after which to stop early.")
    parser.add_argument('--early_stopping_min_delta', type=float, default=10.0,
                        help="Min difference in validation loss to consider non-improving.")
    parser.add_argument('--optim', type=str, default='adam', help='choose the optimizer')
    parser.add_argument('--loss', type=str, default='ce', help='Choose the type of loss function.')

    args = parser.parse_args()
    datasets_params['net'] = models[args.model]

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
