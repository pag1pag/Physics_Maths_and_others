"""
A module to transform an image using conformal mappings.


Inspired by:
https://stackoverflow.com/questions/63121723/creating-a-meshgrid-from-1-1-to-apply-conformal-maps-to-an-image
"""

from pathlib import Path

import cv2
import numpy as np
from solving_dirichlet_problem import f, inv_f

# Path to the mystery image to be transformed.
in_path = Path(__file__).parent / "img" / "mystery_image.png"
# Path to the transformed image.
out_path = Path(__file__).parent / "img" / "output_image.png"

# Reading an image from file.
in_img = np.array(cv2.imread(str(in_path)))

# Taking dimensions of img.
nb_rows, nb_cols = in_img.shape[0], in_img.shape[1]

# Creating vectors of equally spaced points.
x = np.linspace(-1, 1, num=nb_cols, endpoint=True)
y = np.linspace(-1, 1, num=nb_rows, endpoint=True)

# Using the previous vectors to create a matrix of points.
X, Y = np.meshgrid(x, y)


# constants
rot = np.exp(1j * np.pi / 4)
rot_inv = np.exp(-1j * np.pi / 4)


def conformal_mapping(
    X_array: np.ndarray,
    Y_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Confomral mapping of the image.

    You can change this function to any conformal mapping you want.

    Args:
        X_array (np.ndarray): X coordinates of the image.
        Y_array (np.ndarray): Y coordinates of the image.

    Returns:
        tuple[np.ndarray, np.ndarray]: X and Y coordinates of the transformed image.
    """

    W = X_array + 1j * Y_array

    # # please_dont_look_now --> mystery_image
    # Z = f(W**2)

    # mystery_image --> output_image
    Z = np.sqrt(inv_f(W))

    return Z.real, Z.imag


# Applying the conformal mapping.
X, Y = conformal_mapping(X, Y)


def scale_mapping(
    X_array: np.ndarray,
    Y_array: np.ndarray,
    numbers_row: int,
    numbers_column: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts coordinates from a range of [-1,1] to a range of [0,width-1], as needed by cv2.remap

    Args:
        X_array (np.ndarray): X coordinates of the image.
        Y_array (np.ndarray): Y coordinates of the image.
        numbers_row (int): number of rows of the image.
        numbers_column (int): number of columns of the image.

    Returns:
        tuple[np.ndarray, np.ndarray]: X and Y coordinates of the transformed image.
    """
    # Scaling the values to the right range.
    X_new = (X_array / np.max(np.abs(X_array)) + 1) * numbers_column / 2
    Y_new = (Y_array / np.max(np.abs(Y)) + 1) * numbers_row / 2

    # Clipping the values to the right range
    # i.e. values above 0 are set to 0,
    # and values above numbers_column/number are set to numbers_column.
    X_new = np.clip(X_new, 0, numbers_column - 1)
    Y_new = np.clip(Y_new, 0, numbers_row - 1)

    # astype(np.float32) is necessary for cv2.remap to work.
    X_new = np.floor(X_new).astype(np.float32)
    Y_new = np.floor(Y_new).astype(np.float32)
    return X_new, Y_new


# Maps the output of the complex logarithmic mapping to normal pixel coordinates,
# such that cv2.remap can move the pixels to all the right places.
Xnew, Ynew = scale_mapping(X, Y, nb_rows, nb_cols)

# Applying the transformation.
out_img = cv2.remap(in_img, Xnew, Ynew, interpolation=cv2.INTER_CUBIC)

# Saving the transformed image.
cv2.imwrite(str(out_path), out_img)
