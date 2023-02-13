import numpy as np
from scipy.ndimage import interpolation

def apply_transformation(img, transformation):
    # Apply transformation to image
    transformed_image = interpolation.affine_transform(img, transformation, mode="nearest")
    return transformed_image