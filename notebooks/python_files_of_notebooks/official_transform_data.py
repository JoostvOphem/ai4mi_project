import numpy as np
from numpy import pi as π
import nibabel as nib
from pathlib import Path
from scipy.ndimage import affine_transform


# copied from announcement on Canva

#!/usr/bin/env python3.10

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

TR = np.asarray([[1, 0, 0, 50],
                 [0,  1, 0, 40],  # noqa: E241
                 [0,             0,      1, 15],  # noqa: E241
                 [0,             0,      0, 1]])  # noqa: E241

DEG: int = 27
ϕ: float = - DEG / 180 * π
RO = np.asarray([[np.cos(ϕ), -np.sin(ϕ), 0, 0],  # noqa: E241, E201
                 [np.sin(ϕ),  np.cos(ϕ), 0, 0],  # noqa: E241
                 [     0,         0,     1, 0],  # noqa: E241, E201
                 [     0,         0,     0, 1]])  # noqa: E241, E201

X_bar: float = 275
Y_bar: float = 200
Z_bar: float = 0
C1 = np.asarray([[1, 0, 0, X_bar],
                 [0, 1, 0, Y_bar],
                 [0, 0, 1, Z_bar],
                 [0, 0, 0,    1]])  # noqa: E241
C2 = np.linalg.inv(C1)

AFF = C1 @ RO @ C2 @ TR
INV = np.linalg.inv(AFF)

transformation_matrix = INV[:3, :3]

translation_vector = INV[:3, 3]

# iterate over patients
for patient_number in range(1, 41):
    patient_ID = "{:02d}".format(patient_ID)
    print(f"working on patient {patient_ID}")
    fake_nii = nib.load(Path("..") / "data" / "segthor_train" / "train" / f"Patient_{patient_number}" / "GT.nii.gz")
    fake_array = np.array(fake_nii.dataobj)

    # save non-heart organs
    saved_fake_array = np.copy(fake_array)
    saved_fake_array[saved_fake_array == 2] = 0

    # mask with only the heart
    fake_array = (fake_array == 2)

    transformed_array = affine_transform(
        fake_array, 
        transformation_matrix,  # The 3x3 transformation matrix (rotation + scale + shear)
        offset=translation_vector,  # The translation vector
        order=0 # uses nearest-neighbor interpolation
    )
    # re-add the other organs
    saved_fake_array[transformed_array == 1] = 2

    # save the found array
    save_array_ass_nii(saved_fake_array, Path("..") / "data" / "segthor_train" / "train" / f"Patient_{patient_number}" / "real_GT.nii.gz", fake_nii)