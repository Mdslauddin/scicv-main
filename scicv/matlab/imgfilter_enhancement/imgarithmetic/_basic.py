import numpy as np 
"""
def imabsdiff(x,y):
    return np.abs(x,y)

"""
"""__all__ = ['imabsdiff','_imadd','imapplymatrix','imcomplement','imdivide','imlincomb','immultiply','imsubtract']"""

def _imadd(x,y):
    return np.clip(np.add(x,y),0,255)

def imabsdiff(x,y):
    z= np.subtract(x,y)
    return np.abs(z)


def imabs(x,y):
    return np.abs(x,y)


def incomplement(img):
    comimg = np.subtract(255,img)
    return np.clip(comimg,0,255)

def imdivide(x,y):
    divide = np.divide(x,y)
    around = np.around(divide)
    return np.array(around,dtype=np.uint32)


def immultiply(x,y):
    return np.clip(np.multiply(x,y),0,255)


def imsubtract(x,y):
    return np.clip(np.subtract(x,y),0,255)


