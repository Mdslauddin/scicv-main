from ..filters import (gaussian_blur,average_filter,laplacian_filter,sobel_filter,box_blur,add_filter,sub_filter,
multiply_filter,divide_filter,square_filter)


"""
# Design image filter 
    1.) fspecial - create predefine 2-D filte
    2.) fspecial - create predefine 3-D filter 
    3.) convmtx2 - 2-D convolution matrix 




"""
__all__ = ['imfilter']

# N-D filter of multidimension image
def imfilter(img,method='gaussian'):
    """
    
    method list:
    1.) average # Averaging filter
    2.) disk # Circular averaging filter (pillbox)
    3.) gaussian # Gaussian lowpass filter.
    4.) laplacian # Approximates the two-dimensional Laplacian operator
    5.) motion # Approximates the linear motion of a camera
    6.) prewitt # Prewitt horizontal edge-emphasizing filter
    7.) sobel # Sobel horizontal edge-emphasizing filter
    8.) log # Laplacian of Gaussian filter
    9.) boxblur
    10.) add_filter
    11.) sub_filter
    12.) multiply_filter
    13.) divide_filter
    14.) square_filter

    
    """
    if (method == 'average'):
        kernel_size = int(input("Enter kernel_size: "))
        return average_filter(img,kernel_size)
    
    elif (method == 'disk'):
        return 0

    elif (method == 'gaussian'):
        sigma = int(input("Enter Sigma Value: "))
        return gaussian_blur(img,sigma)
    elif (method =='laplacian'):
        ksize = int(input("Kernel Size : "))
        return laplacian_filter(img,ksize) 
    elif (method == 'motion'):
        return 0 

    elif (method =='prewitt'):
        return 0 
    
    elif (method == 'sobel'):
        return sobel_filter(img)
    elif (method == 'log'):
        return 0 

    elif( method == 'add_filter'):
        x = int(input("Enter Value of X : "))
        return add_filter(img,x)

    elif (method == "sub_filter"):
        x = int(input("Enter Value of X : "))
        return sub_filter(img,x)

    elif (method == 'multiply_filter'):
        x = int(input("Enter Value of X : "))
        return multiply_filter(img,x)

    elif (method == 'divide_filter'):
        x = int(input("Enter Value of X : "))
        return divide_filter(img,x)

    elif (method == 'square_filter'):
        return square_filter(img)

    else:
        return ("Wrong method ")
    

"""
from scipy.signal import convolve2d

def motion_filter(img, kernel):
    # Use convolve2d function to convolve the image with the kernel
    filtered_img = convolve2d(img, kernel, mode='same')
    return filtered_img

# Example usage
img =cmeraman# cv2.imread("image.jpg",0)
# Create a motion kernel
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
filtered_img = motion_filter(img, kernel)


"""