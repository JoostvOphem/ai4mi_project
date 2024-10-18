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

import torch
from torch import einsum, Tensor

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor) -> Tensor:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        # Use '...' in einsum to handle both 2D and 3D data
        loss = - einsum("bk...,bk...->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss

class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)

class DiceLoss:
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.smooth = 1e-5
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor) -> Tensor:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        pred = pred_softmax[:, self.idk, ...]
        target = weak_target[:, self.idk, ...].float()

        intersection = (pred * target).sum(dim=(2, 3, 4) if pred.dim() == 5 else (2, 3))
        union = pred.sum(dim=(2, 3, 4) if pred.dim() == 5 else (2, 3)) + target.sum(dim=(2, 3, 4) if pred.dim() == 5 else (2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()

        return loss
    
class FocalLoss:
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.gamma = kwargs.get('gamma', 2.0)
        self.alpha = kwargs.get('alpha', 0.25)
        self.smooth = 1e-6
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor) -> Tensor:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        pred = pred_softmax[:, self.idk, ...]
        target = weak_target[:, self.idk, ...].float()

        # Compute focal loss
        pt = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma

        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = -alpha_t * focal_weight * torch.log(pt + self.smooth)

        # Average over non-ignored pixels
        num_non_zero = torch.sum(target > 0) + self.smooth
        return torch.sum(loss) / num_non_zero
    
class CombinedLoss:
    def __init__(self, **kwargs):
        self.ce_loss = CrossEntropy(**kwargs)
        self.dice_loss = DiceLoss(**kwargs)
        self.alpha = kwargs.get('alpha', 0.5)  # Weight balance between CE and Dice
        self.idk = kwargs['idk']  # Classes to consider
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax: Tensor, target: Tensor) -> Tensor:
        assert pred_softmax.shape == target.shape
        assert simplex(pred_softmax)
        assert sset(target, [0, 1])

        ce_loss = self.ce_loss(pred_softmax, target)
        dice_loss = self.dice_loss(pred_softmax, target)

        combined_loss = self.alpha * ce_loss + (1 - self.alpha) * dice_loss

        return combined_loss

    def dice_coefficient(self, pred_softmax: Tensor, target: Tensor) -> Tensor:
        # Calculate Dice coefficient for monitoring
        pred = pred_softmax[:, self.idk, ...]
        target = target[:, self.idk, ...].float()

        intersection = (pred * target).sum(dim=(2, 3, 4) if pred.dim() == 5 else (2, 3))
        union = pred.sum(dim=(2, 3, 4) if pred.dim() == 5 else (2, 3)) + target.sum(dim=(2, 3, 4) if pred.dim() == 5 else (2, 3))

        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        return dice.mean()