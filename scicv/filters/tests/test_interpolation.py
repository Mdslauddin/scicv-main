__all__ = ['affine_transform', 'geometric_transformation', 'map_coordinate', 'rotate', 'shift', 
          'splines_filter', 'splines_filter1d', 'zoom']

import numpy as np



import numpy as np

def affine_transform(points, matrix):
    """
    # Define the translation and rotation matrices
    tx = 2.0
    ty = 1.0
    theta = np.pi/4
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])

    # Define a set of points
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    # Apply the affine transformation
    transformed_points = apply_affine_transform(points, np.dot(translation_matrix, rotation_matrix))

    print(transformed_points)

    
    Applies an affine transformation to a set of 2D points.

    Parameters:
    - points: a numpy array of shape (n, 2) containing n 2D points
    - matrix: a 3x3 numpy array representing the affine transformation matrix

    Returns:
    - a numpy array of shape (n, 2) containing the transformed points
    """
    # Add a homogeneous coordinate of 1 to each point
    n = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((n, 1))))

    # Apply the transformation matrix
    transformed_points = np.dot(homogeneous_points, matrix.T)

    # Divide by the last coordinate to obtain the transformed points
    transformed_points[:, 0] /= transformed_points[:, 2]
    transformed_points[:, 1] /= transformed_points[:, 2]

    # Remove the homogeneous coordinate and return the result
    return transformed_points[:, :2]


import numpy as np

def geometric_transform(points, transform):
    """
    # Define the perspective transform matrix
    transform = np.array([[1, 0.2, 0], [0.2, 1, 0], [0.1, 0.1, 1]])

    # Define a set of points
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    # Apply the geometric transform
    transformed_points = apply_geometric_transform(points, transform)

    print(transformed_points)

    
    Applies an arbitrary geometric transform to a set of 2D points.

    Parameters:
    - points: a numpy array of shape (n, 2) containing n 2D points
    - transform: a 3x3 numpy array representing the transform matrix

    Returns:
    - a numpy array of shape (n, 2) containing the transformed points
    """
    # Add a homogeneous coordinate of 1 to each point
    n = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((n, 1))))

    # Apply the transformation matrix
    transformed_points = np.dot(homogeneous_points, transform.T)

    # Divide by the last coordinate to obtain the transformed points
    transformed_points[:, 0] /= transformed_points[:, 2]
    transformed_points[:, 1] /= transformed_points[:, 2]

    # Remove the homogeneous coordinate and return the result
    return transformed_points[:, :2]



import numpy as np
from scipy.interpolate import RegularGridInterpolator

def map_coordinate(input_array, old_coordinates, new_coordinates):
    """
    map_array_to_new_coordinates
    Maps an input array to new coordinates using interpolation.

    Parameters:
    - input_array: a numpy array of shape (n, m) containing the input data
    - old_coordinates: a tuple of n 1D numpy arrays containing the old coordinates
    - new_coordinates: a tuple of n 1D numpy arrays containing the new coordinates

    Returns:
    - a numpy array of shape (p, q) containing the mapped data
    """
    # Create a regular grid interpolator for the input data
    interpolator = RegularGridInterpolator(old_coordinates, input_array)

    # Evaluate the interpolator at the new coordinates
    mapped_array = interpolator(new_coordinates)

    return mapped_array


def rotate(arr, angle):
    """Rotate a 2D array by a given angle in degrees.
    
    Args:
        arr (ndarray): The input array to rotate.
        angle (float): The angle to rotate the array in degrees.
    
    Returns:
        The rotated array.
    """
    # Convert the angle from degrees to radians
    angle_rad = np.radians(angle)
    
    # Get the shape of the input array
    rows, cols = arr.shape
    
    # Calculate the coordinates of the center of the array
    cx = cols / 2
    cy = rows / 2
    
    # Create a rotation matrix
    rot_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad), np.cos(angle_rad)]])
    
    # Create a grid of coordinates for the input array
    xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Shift the grid of coordinates so that the origin is at the center of the array
    xx -= cx
    yy -= cy
    
    # Apply the rotation matrix to the grid of coordinates
    new_coords = rot_mat.dot(np.vstack([xx.flatten(), yy.flatten()]))
    
    # Reshape the coordinates back into a grid
    new_xx = np.reshape(new_coords[0, :], (rows, cols))
    new_yy = np.reshape(new_coords[1, :], (rows, cols))
    
    # Shift the grid of coordinates back to the original origin
    new_xx += cx
    new_yy += cy
    
    # Interpolate the rotated array from the original array using the new coordinates
    rotated_arr = np.interp(new_yy.flatten(), np.arange(rows), arr[:, ::-1].T).reshape(arr.shape)
    rotated_arr = np.interp(new_xx.flatten(), np.arange(cols), rotated_arr[::-1, :]).reshape(arr.shape)
    
    return rotated_arr


import numpy as np

def shift(input_array, shift_x, shift_y):
    """
    
    # Define the input array
    input_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Shift the input array by 1.5 pixels in the x direction and 2.5 pixels in the y direction
    shifted_array = shift_array(input_array, 1.5, 2.5)

    print(shifted_array)

    Shifts a 2D input array by a specified amount in the x and y directions.

    Parameters:
    - input_array: a numpy array of shape (n, m) containing the input data
    - shift_x: a float specifying the amount to shift the array in the x direction
    - shift_y: a float specifying the amount to shift the array in the y direction

    Returns:
    - a numpy array of shape (n, m) containing the shifted data
    """
    # Define the shifted coordinates
    shifted_x = np.arange(input_array.shape[1]) + shift_x
    shifted_y = np.arange(input_array.shape[0]) + shift_y

    # Use numpy's meshgrid function to create a grid of the shifted coordinates
    grid_shifted_x, grid_shifted_y = np.meshgrid(shifted_x, shifted_y, indexing='ij')

    # Use scipy's interpolation function to interpolate the input array at the shifted coordinates
    from scipy.interpolate import RectBivariateSpline
    interpolator = RectBivariateSpline(np.arange(input_array.shape[0]), np.arange(input_array.shape[1]), input_array)
    shifted_array = interpolator(grid_shifted_y, grid_shifted_x)

    return shifted_array


import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import convolve

def spline_filter(input_array, order=3, output_shape=None):
    """
    # Define the input array
    input_array = np.array([[0, 0, 0, 0, 0],
                        [0, 1, 2, 1, 0

    Applies a multidimensional spline filter to a multidimensional input array.

    Parameters:
    - input_array: a numpy array containing the input data
    - order: an integer specifying the order of the spline interpolation (default: 3)
    - output_shape: a tuple specifying the shape of the output array (default: None)

    Returns:
    - a numpy array containing the filtered data
    """
    # Create a RectBivariateSpline object for each dimension of the input array
    splines = []
    for dim in range(input_array.ndim):
        spline = RectBivariateSpline(np.arange(input_array.shape[dim]), np.arange(input_array.shape[-1]), input_array.swapaxes(dim, -1), kx=order, ky=order)
        splines.append(spline)

    # Create a grid of points for each dimension of the output array
    if output_shape is None:
        output_shape = input_array.shape
    output_grids = []
    for dim in range(input_array.ndim):
        output_grid = np.linspace(0, input_array.shape[dim] - 1, output_shape[dim])
        output_grids.append(output_grid)

    # Use the RectBivariateSpline objects to interpolate the input array to the output grids
    output_array = input_array
    for dim in range(input_array.ndim):
        output_array = splines[dim](output_grids[dim], np.arange(output_array.shape[-1])).swapaxes(dim, -1)

    # Normalize the output array to have a maximum value of 1
    output_array /= np.max(output_array)

    # Apply a Gaussian smoothing filter to the output array
    smoothing_filter = np.zeros((3,) * input_array.ndim)
    smoothing_filter[(1,) * input_array.ndim] = 1
    smoothing_filter = convolve(smoothing_filter, np.ones((3,) * input_array.ndim), mode='constant')
    output_array = convolve(output_array, smoothing_filter, mode='constant')

    return output_array

import numpy as np
from scipy.interpolate import splrep, splev

def spline_filter1d(input_array, axis=0, order=3, output_shape=None):
    """
    # Define the input array
    input_array = np.array([[0, 0, 0, 0, 0],
                        [0, 1, 2, 1, 0],
                        [0, 2, 4, 2, 0],
                        [0, 1, 2, 1, 0],
                        [0, 0, 0, 0, 0]])

    # Filter the input array along the y-axis using a 1-D spline filter of order 3
    filtered_array = spline_filter_1d(input_array, axis=0, order=3)

    # Print the filtered array
    print(filtered_array)

    
    Calculates a 1-D spline filter along the specified axis of a multidimensional input array.

    Parameters:
    - input_array: a numpy array containing the input data
    - axis: an integer specifying the axis along which to filter the data (default: 0)
    - order: an integer specifying the order of the spline interpolation (default: 3)
    - output_shape: a tuple specifying the shape of the output array (default: None)

    Returns:
    - a numpy array containing the filtered data
    """
    # Create a 1-D spline representation of the input array along the specified axis
    tck = splrep(np.arange(input_array.shape[axis]), input_array, k=order, axis=axis)

    # Evaluate the 1-D spline representation at the specified output points
    if output_shape is None:
        output_shape = input_array.shape
    output_points = np.indices(output_shape).reshape((input_array.ndim, -1))
    output_points[axis, :] = np.arange(output_shape[axis])
    output_array = splev(output_points[axis, :], tck)

    return output_array


import numpy as np
from scipy.ndimage import zoom

def zoom(input_array, zoom_factors):
    """
    # Define the input array
    input_array = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

    # Zoom the input array by a factor of 2 along each axis
    zoom_factors = (2, 2)
    zoomed_array = zoom_array(input_array, zoom_factors)

    # Print the zoomed array
    print(zoomed_array)

    
    Zooms a multidimensional input array by the specified factors along each axis.

    Parameters:
    - input_array: a numpy array containing the input data
    - zoom_factors: a tuple containing the zoom factors for each axis

    Returns:
    - a numpy array containing the zoomed data
    """
    # Calculate the output shape based on the input shape and zoom factors
    input_shape = input_array.shape
    output_shape = tuple(int(round(input_shape[i] * zoom_factors[i])) for i in range(input_array.ndim))

    # Zoom the input array to the specified output shape
    output_array = zoom(input_array, zoom_factors, order=3)

    return output_array


"""
import numpy as np

def zoom_array(arr, zoom_factor):
    """

    # Define the input array
    input_array = np.array([[1, 2, 3],
                        [

    Zooms a numpy array by a given factor using bilinear interpolation.

    Parameters:
    - arr: input numpy array
    - zoom_factor: zooming factor as a float

    Returns:
    - numpy array containing the zoomed data
    """
    # Get the dimensions of the input array
    h, w = arr.shape[:2]

    # Calculate the new dimensions of the output array after zooming
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)

    # Create the output array with the new dimensions
    output = np.zeros((new_h, new_w) + arr.shape[2:], dtype=arr.dtype)

    # Calculate the scaling factor for each dimension
    sh, sw = 1 / zoom_factor, 1 / zoom_factor

    # Create a grid of coordinates for the output array
    r, c = np.mgrid[:new_h, :new_w]

    # Calculate the corresponding coordinates in the input array
    y, x = r * sh, c * sw

    # Calculate the integer coordinates for each point
    x0, y0 = np.floor(x).astype(np.int), np.floor(y).astype(np.int)
    x1, y1 = x0 + 1, y0 + 1

    # Make sure the integer coordinates are within the bounds of the input array
    x0, x1 = np.clip(x0, 0, w-1), np.clip(x1, 0, w-1)
    y0, y1 = np.clip(y0, 0, h-1), np.clip(y1, 0, h-1)

    # Calculate the weights for each point based on the distance to the integer coordinates
    wx1, wx0 = x - x0, 1 - (x - x0)
    wy1, wy0 = y - y0, 1 - (y - y0)

    # Interpolate the values for each point using bilinear interpolation
    output[r, c] = wx0 * wy0 * arr[y0, x0] + wx1 * wy0 * arr[y0, x1] + wx0 * wy1 * arr[y1, x0] + wx1 * wy1 * arr[y1, x1]

    return output

"""