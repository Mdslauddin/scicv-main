
"""
def imabsdiff(x,y):
    return np.abs(x,y)

"""
import numpy as np 

__all__ = ['imadd','imsubtract','immultiply','imdivide','imsquare','padarray','imabsdiff','imabs','incomplement']

def imadd(x,y):
    return np.clip(np.add(x,y),0,255)

def imsubtract(x,y):
    return np.clip(np.subtract(x,y),0,255)

def immultiply(x,y):
    return np.clip(np.multiply(x,y),0,255)

def imdivide(x,y):
    divide = np.divide(x,y)
    around = np.around(divide)
    return np.array(around,dtype=np.uint32)

def imsquare(image):
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



def imabsdiff(x,y):
    z= np.subtract(x,y)
    return np.abs(z)


def imabs(x,y):
    return np.abs(x,y)


def incomplement(img):
    comimg = np.subtract(255,img)
    return np.clip(comimg,0,255)









