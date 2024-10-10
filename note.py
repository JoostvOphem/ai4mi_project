import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# orginal
original = nib.load("data/segthor_train/train/Patient_03/Patient_03.nii.gz")
original = np.asarray(original.dataobj)
x = original.shape[0]
plt.imshow(original[int(x / 2), :, :], cmap='gray')
plt.savefig("original.png")


# UNETR data
unetr = nib.load("data/SEGTHOR_3D/train/img/Patient_03.nii.gz")
unetr = np.asarray(unetr.dataobj)
x = unetr.shape[0]
plt.imshow(unetr[int(x / 2), :, :], cmap='gray')
plt.savefig("unetr.png")