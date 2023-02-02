import numpy as np 

__all__ = ['add_filter','sub_filter','multiply_filter','divide_filter','square_filter']
 

def add_filter(image,x):
    return np.add(x,image)


def sub_filter(image,x):
    return np.subtract(image,x)

def multiply_filter(image,x):
    return np.multiply(x,image)


def divide_filter(image,x):
    return np.divide(x,image)

def square_filter(image):
    return np.square(image)

def meanStdDev(img):
    """
    # Calculate the mean and standard deviation of the pixel values
    mean, std_dev = meanStdDev(gray)
    
    """
    # Calculate the mean and standard deviation of the pixel values
    mean = np.mean(img)
    std_dev = np.std(img)
    # Apply the standard deviation filter to the image
    img_filtered = np.clip(gray - mean + std_dev, 0, 255).astype(np.uint8)
    return img_filtered

def padarray(img):
    # Define the padding widths
    pad_width = [(1, 1), (1, 1)]
    # Pad the array using np.pad
    padded_array = np.pad(img, pad_width, mode='constant', constant_values=0)
    return padded_array



