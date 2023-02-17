__all__ = ['fourier_ellipsoid','fourier_gaussian','fourier_shift','fourier_uniform']

import numpy as np
from scipy.fftpack import fftn, ifftn

def fourier_ellipsoid(input_array, radii, center=None):
    """Apply an ellipsoid Fourier filter to a multidimensional array.
    
    Args:
        input_array (ndarray): The input array to filter.
        radii (tuple): The radii of the ellipsoid in each dimension.
        center (tuple, optional): The center of the ellipsoid. If not specified,
            the center is assumed to be at the center of the input array.
    
    Returns:
        The filtered array.
    """
    # Get the shape of the input array
    shape = input_array.shape
    
    # Calculate the center of the input array if not specified
    if center is None:
        center = tuple([s//2 for s in shape])
    
    # Create a coordinate grid with the same shape as the input array
    coords = np.indices(shape)
    
    # Shift the coordinates to the center of the array
    coords = coords - np.array(center)[:, np.newaxis, np.newaxis]
    
    # Calculate the squared distance from the center for each point in the array
    sq_dist = np.sum((coords/radii)**2, axis=0)
    
    # Create a boolean mask for points inside the ellipsoid
    mask = sq_dist <= 1
    
    # Apply the mask to the Fourier transform of the input array
    input_fft = fftn(input_array)
    filtered_fft = input_fft * mask
    filtered_array = np.real(ifftn(filtered_fft))
    
    return filtered_array



def fourier_gaussian(input_array, sigma):
    """Apply a Gaussian Fourier filter to a multidimensional array.
    
    Args:
        input_array (ndarray): The input array to filter.
        sigma (float or tuple): The standard deviation of the Gaussian filter in each dimension.
            If a float is given, the same value is used for all dimensions.
    
    Returns:
        The filtered array.
    """
    # Get the shape of the input array
    shape = input_array.shape
    
    # Calculate the Fourier transform of the input array
    input_fft = fftn(input_array)
    
    # Create a coordinate grid with the same shape as the input array
    coords = np.indices(shape)
    
    # Shift the coordinates to the center of the array
    center = tuple([s//2 for s in shape])
    coords = coords - np.array(center)[:, np.newaxis, np.newaxis]
    
    # Calculate the squared distance from the center for each point in the array
    sq_dist = np.sum((coords**2), axis=0)
    
    # Calculate the Gaussian filter in Fourier space
    if isinstance(sigma, float):
        sigma = (sigma,) * len(shape)
    filter_fft = np.exp(-sq_dist / (2 * np.array(sigma)**2))
    
    # Apply the Gaussian filter to the Fourier transform of the input array
    filtered_fft = input_fft * filter_fft
    filtered_array = np.real(ifftn(filtered_fft))
    
    return filtered_array




def fourier_shift(input_array, shifts):
    """Apply a Fourier shift filter to a multidimensional array.
    
    Args:
        input_array (ndarray): The input array to filter.
        shifts (tuple): The amount of shift in each dimension.
    
    Returns:
        The filtered array.
    """
    # Get the shape of the input array
    shape = input_array.shape
    
    # Calculate the Fourier transform of the input array
    input_fft = fftn(input_array)
    
    # Create a coordinate grid with the same shape as the input array
    coords = np.indices(shape)
    
    # Calculate the phase shift for each point in the array
    phase_shift = np.exp(2j * np.pi * np.sum(coords * np.array(shifts)[:, np.newaxis, np.newaxis] / np.array(shape)[:, np.newaxis, np.newaxis], axis=0))
    
    # Apply the phase shift to the Fourier transform of the input array
    filtered_fft = input_fft * phase_shift
    filtered_array = np.real(ifftn(filtered_fft))
    
    return filtered_array


def fourier_uniform(input_array, radius):
    """Apply a uniform Fourier filter to a multidimensional array.
    
    Args:
        input_array (ndarray): The input array to filter.
        radius (float): The radius of the filter in Fourier space.
    
    Returns:
        The filtered array.
    """
    # Get the shape of the input array
    shape = input_array.shape
    
    # Calculate the Fourier transform of the input array
    input_fft = fftn(input_array)
    
    # Create a coordinate grid with the same shape as the input array
    coords = np.indices(shape)
    
    # Shift the coordinates to the center of the array
    center = tuple([s//2 for s in shape])
    coords = coords - np.array(center)[:, np.newaxis, np.newaxis]
    
    # Calculate the squared distance from the center for each point in the array
    sq_dist = np.sum((coords**2), axis=0)
    
    # Create a mask for points within the radius
    mask = np.where(sq_dist <= radius**2, 1, 0)
    
    # Apply the mask to the Fourier transform of the input array
    filtered_fft = input_fft * mask
    filtered_array = np.real(ifftn(filtered_fft))
    
    return filtered_array

