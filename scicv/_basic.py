import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

__all__ = ['gauss_filter_3d','lowpass_filter_3d']

def gauss_filter_3d(volume, sigma):
    # Create a 3D Gaussian kernel
    kernel_size = int(6*sigma + 1)
    kernel = np.zeros((kernel_size, kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                x = i - center
                y = j - center
                z = k - center
                kernel[i, j, k] = np.exp(-(x**2 + y**2 + z**2)/(2*sigma**2))
    kernel /= kernel.sum()
    
    # Convolve the volume with the kernel
    return convolve(volume, kernel)



def lowpass_filter_3d(volume, kernel_size):
    # Create a 3D low-pass kernel
    kernel = np.ones((kernel_size, kernel_size, kernel_size))
    kernel /= kernel.size
    
    # Convolve the volume with the kernel
    return convolve(volume, kernel)


def highpass_filter_3d(volume, kernel_size):
    # Create a 3D high-pass kernel
    kernel = np.ones((kernel_size, kernel_size, kernel_size))
    kernel /= kernel.size
    center = kernel_size // 2
    kernel[center, center, center] = -(kernel.size - 1)
    
    # Convolve the volume with the kernel
    return convolve(volume, kernel)
