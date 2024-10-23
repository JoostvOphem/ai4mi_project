import skimage.morphology as mrph
from src.modules.load_segmentation import to_onehot_tensor
from pathlib import Path
import numpy as np
from skimage.measure import label

# https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image
def getLargestCC(segmentation):
    labels = label(segmentation)
    if labels.max() == 0:
        return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def closing(volume, radius):
    footprint = mrph.ball(radius)
    channels = [t.numpy().T for t in to_onehot_tensor(volume)[0][1:]]
    new_img = np.zeros(channels[0].shape).astype(np.uint8)

    i = 1
    for c in channels:
        c = getLargestCC(mrph.binary_closing(c, footprint))
        new_img[c] = i
        i += 1
    return new_img

def opening(volume, radius):
    footprint = mrph.ball(radius)
    channels = [t.numpy().T for t in to_onehot_tensor(volume)[0][1:]]
    new_img = np.zeros(channels[0].shape).astype(np.uint8)

    i = 1
    for c in channels:
        c = getLargestCC(mrph.binary_opening(c, footprint))
        new_img[c] = i
        i += 1
    return new_img


def closing_opening(volume, closing_radius, opening_radius):
    closing_footprint = mrph.ball(closing_radius)
    opening_footprint = mrph.ball(opening_radius)
    channels = [t.numpy().T for t in to_onehot_tensor(volume)[0][1:]]
    new_img = np.zeros(channels[0].shape).astype(np.uint8)

    i = 1
    for c in channels:
        c = mrph.binary_closing(c, closing_footprint)
        c = getLargestCC(mrph.binary_opening(c, opening_footprint))
        new_img[c] = i
        i += 1
    return new_img