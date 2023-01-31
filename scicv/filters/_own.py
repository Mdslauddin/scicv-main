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


