import numpy as np 

__all__ = ['cvtColor']

def cvtColor(img,color):
   
    if color == 'COLOR_BGR2GRAY':
        # Convert an image to grayscale 
        return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])# return grayscale imag