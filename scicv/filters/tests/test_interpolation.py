__all__ = ['rotate']

import numpy as np

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
