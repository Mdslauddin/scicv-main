"""
# Image filtering: Convolution and correlation predefind and custom filters, nonlinear filter,edge-prevision filter 

# Contrast Adjustment : Contrast adjustment, histogram equalization, decorrelation stretching

# ROI : Define and operate on regions of interest (ROI)

# Morphological Operations : Dilate, erode, reconstruct, and perform other morphological operations

# Deblurring : Deconvolution for deblurring

# Neighborhood and Block Processing : Define neighborhoods and blocks for filtering and I/O operations

# Image Arithmetic : Add, subtract, multiply, and divide images




## Desing image filters
- 'fspecail'
- 'fspecial3'
- 'convmtx2' 

## Basic image filtering in the Spatial Domain
- 'imfilter'
- 'roifilt2'
- 'nlfilter'
- 'imgaussfilt'
- 'imgaussfilt3'
- 'wiener2','medfilt2'
- 'medfilt3'
- 'modefilt'
- 'ordfilt2'
- 'stdfilt'
- 'rangefilt'
- 'entropyfilt'
- 'imboxfilt'
- 'imboxfilt3'
- 'fibermetric'
- 'maxhessiannorm'
- 'padarray'

## Edge Preserning filtering
 - 'imbilatfilt'
 - 'imdiffuseest'
 - 'imguidedfilter'
 - 'imnlmfilt'
 - 'burstinterpolant'


## Texture filtering
- 'gabor'
- 'imgaborfilt'


## Filtering by Property characteristic
 - 'bwarefilt'
 - 'bwpropfilt'

## Intergral Image Domain Filtering 

 - 'integralImage'
 - 'intergralImage3'
 - 'intergralBoxfilter'
 - 'intergralBoxfilter'

## Frequency Domain Filter 

 - freqspace
 - freqz2
 - fsam2
 - ftrans2
 - fwid
 - fwind 



"""

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from cv2 import bilateralFilter
import cv2
from scipy.signal import convolve2d
from PIL import Image
import matplotlib.pyplot as plt



__all__ = ['box_blur','gaussian_blur','median_blur','bilateralFilter',
 
          'kuwahara_filter','laplacian_filter','unsharp_mask','sobel_filter',
          'average_filter', 'gauss_filter_3d', 'lowpass_filter_3d', 'highpass_filter_3d'
        
        ]



"""

__all__ = ['imgaussfilt','imgaussfilt3','wiener2','medfilt2','medfilt3','modefilt','ordfilt2','stdfilt','rangefilt',
 'entropyfilt','imboxfilt','imboxfilt3','fibermetric','maxhessiannorm','padarray']




"""

# Box blur 
def box_blur(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    kernel = np.array(kernel,ndmin=3)
    return convolve(image, kernel)

# Gaussian blur 
def gaussian_blur(image, sigma=35):
    return gaussian_filter(image, sigma)
     
# Median blur
def median_blur(image, kernel_size=13):
    return median_filter(image, size=kernel_size)


def bilateral_blur(image, d=9, sigmaColor=75, sigmaSpace=75):
    return bilateralFilter(image, d, sigmaColor, sigmaSpace)

def kuwahara_filter(image, kernel_size=3):
    return cv2.kuwahara(image, kernel_size)

def laplacian_filter(image, ksize=3):
    return cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)




def unsharp_mask(image, kernel_size=3, strength=1):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    blur = convolve(image, kernel)
    return image + strength * (image - blur)


def sobel_filter(image, xorder=1, yorder=1, ksize=3):
    return cv2.Sobel(image, cv2.CV_64F, xorder, yorder, ksize=ksize)


def average_filter(img, kernel_size):
    # Create a kernel of size kernel_size x kernel_size filled with 1/(kernel_size*kernel_size)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)/(kernel_size*kernel_size)
    # Use convolve2d function to convolve the image with the kernel
    filtered_img = convolve2d(img, kernel, mode='same', boundary='symm')
    return filtered_img






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
