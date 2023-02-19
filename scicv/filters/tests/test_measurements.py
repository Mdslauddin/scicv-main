



__all__ = ['center_of_mass',  'variance', 'watershed_ift' ]

import numpy as np

def center_of_mass(arr, labels):
    """
    center_of_mass_at_labels
    """
    unique_labels = np.unique(labels)
    center_of_mass = np.zeros_like(unique_labels, dtype=float)
    
    for i, label in enumerate(unique_labels):
        indices = np.where(labels == label)
        values = arr[indices]
        mass = np.sum(values)
        center_of_mass[i] = np.sum(values * indices) / mass
    
    return center_of_mass


import numpy as np
from skimage.morphology import watershed
from skimage.graph import route_through_array

def watershed_ift(img, markers):
    """
    watershed_from_markers
    """
    # Apply the watershed transform using the markers as seeds
    ws = watershed(-img, markers, mask=img)
    
    # Calculate the image foresting transform
    f = img.max() - img
    
    # Calculate the shortest paths from the markers to the background
    paths = [route_through_array(f, markers[i], np.argwhere(ws == 0)[0])
             for i in range(len(np.unique(markers))-1)]
    
    # Create a new marker image from the paths
    markers_new = np.zeros_like(img, dtype=np.int32)
    for i in range(len(paths)):
        path = paths[i]
        for j in range(len(path)):
            markers_new[path[j][0], path[j][1]] = i + 1
    
    # Apply the watershed transform again using the new markers
    ws = watershed(-img, markers_new, mask=img)
    
    return ws

import numpy as np

def variance(img, regions=None):
    """
    image_variance
    """
    if regions is None:
        # If no regions are specified, calculate the variance of the whole image
        return np.var(img)
    else:
        # If regions are specified, calculate the variance of each region separately
        if isinstance(regions, tuple):
            # If a single region is specified as a tuple, convert it to a list
            regions = [regions]
        variances = []
        for region in regions:
            # Extract the sub-image corresponding to the current region
            sub_img = img[region]
            # Calculate the variance of the sub-image and append it to the list of variances
            variances.append(np.var(sub_img))
        return variances

import numpy as np

def value_indices(arr):
    """
    find_indices_of_distinct_values
    """
    # Find the unique values in the array
    unique_values = np.unique(arr)
    
    # Initialize a dictionary to store the indices of each unique value
    indices_dict = {}
    
    # Iterate over the unique values and find the indices of each one
    for val in unique_values:
        indices = np.argwhere(arr == val)
        indices_dict[val] = indices
    
    return indices_dict


import numpy as np

def sum_labels(arr):
    """
    calculate_array_sum
    """
    return np.sum(arr)


import numpy as np

def standard_deviation(img, regions=None):
    """
    image_std_dev
    """
    if regions is None:
        # If no regions are specified, calculate the standard deviation of the whole image
        return np.std(img)
    else:
        # If regions are specified, calculate the standard deviation of each region separately
        if isinstance(regions, tuple):
            # If a single region is specified as a tuple, convert it to a list
            regions = [regions]
        std_devs = []
        for region in regions:
            # Extract the sub-image corresponding to the current region
            sub_img = img[region]
            # Calculate the standard deviation of the sub-image and append it to the list of standard deviations
            std_devs.append(np.std(sub_img))
        return std_devs

    
import numpy as np

def minimum_position(arr, labels):
    """
    find_min_positions_at_labels
    """
    # Find the unique labels in the array
    unique_labels = np.unique(labels)
    
    # Initialize an array to store the positions of the minimums for each label
    min_positions = np.zeros(unique_labels.shape, dtype=int)
    
    # Iterate over the unique labels and find the positions of the minimums for each one
    for i, label in enumerate(unique_labels):
        # Find the positions where the current label occurs in the array
        label_positions = np.argwhere(labels == label)
        
        # Find the minimum value for the current label
        min_val = np.min(arr[label_positions])
        
        # Find the positions where the minimum value occurs for the current label
        min_positions_for_label = label_positions[np.argwhere(arr[label_positions] == min_val)]
        
        # Store the first minimum position for the current label in the output array
        min_positions[i] = min_positions_for_label[0][0]
    
    return min_positions



import numpy as np

def minimum(arr, labels):
    """
    min_over_labeled_regions
    """
    # Find the unique labels in the array
    unique_labels = np.unique(labels)
    
    # Initialize an array to store the minimums for each label
    min_vals = np.zeros(unique_labels.shape)
    
    # Iterate over the unique labels and find the minimum value for each one
    for i, label in enumerate(unique_labels):
        # Find the positions where the current label occurs in the array
        label_positions = np.argwhere(labels == label)
        
        # Find the minimum value for the current label
        min_val = np.min(arr[label_positions])
        
        # Store the minimum value for the current label in the output array
        min_vals[i] = min_val
    
    return min_vals


import numpy as np

def median(arr, labels):
    """
    median_over_labeled_regions
    """
    # Find the unique labels in the array
    unique_labels = np.unique(labels)
    
    # Initialize an array to store the medians for each label
    median_vals = np.zeros(unique_labels.shape)
    
    # Iterate over the unique labels and find the median value for each one
    for i, label in enumerate(unique_labels):
        # Find the positions where the current label occurs in the array
        label_positions = np.argwhere(labels == label)
        
        # Find the median value for the current label
        median_val = np.median(arr[label_positions])
        
        # Store the median value for the current label in the output array
        median_vals[i] = median_val
    
    return median_vals


import numpy as np

def mean(arr, labels):
    """
    mean_over_labeled_regions
    """
    # Find the unique labels in the array
    unique_labels = np.unique(labels)
    
    # Initialize an array to store the means for each label
    mean_vals = np.zeros(unique_labels.shape)
    
    # Iterate over the unique labels and find the mean value for each one
    for i, label in enumerate(unique_labels):
        # Find the positions where the current label occurs in the array
        label_positions = np.argwhere(labels == label)
        
        # Find the mean value for the current label
        mean_val = np.mean(arr[label_positions])
        
        # Store the mean value for the current label in the output array
        mean_vals[i] = mean_val
    
    return mean_vals



import numpy as np

def maximum_position(arr, labels):
    """
    max_positions_over_labeled_regions
    """
    # Find the unique labels in the array
    unique_labels = np.unique(labels)
    
    # Initialize an array to store the positions of the maximums for each label
    max_positions = []
    
    # Iterate over the unique labels and find the position of the maximum value for each one
    for label in unique_labels:
        # Find the positions where the current label occurs in the array
        label_positions = np.argwhere(labels == label)
        
        # Find the maximum value for the current label
        max_val = np.max(arr[label_positions])
        
        # Find the positions of the maximum value for the current label
        max_val_positions = np.argwhere(arr == max_val)
        
        # Filter the positions to only include those that correspond to the current label
        max_label_positions = max_val_positions[np.isin(labels[max_val_positions], label)]
        
        # Add the positions to the output list
        max_positions.append(max_label_positions)
    
    return max_positions



import numpy as np

def maximum(arr, labels):
    """
    max_over_labeled_regions
    """
    # Find the unique labels in the array
    unique_labels = np.unique(labels)
    
    # Initialize an array to store the maximums for each label
    max_vals = np.zeros(unique_labels.shape)
    
    # Iterate over the unique labels and find the maximum value for each one
    for i, label in enumerate(unique_labels):
        # Find the positions where the current label occurs in the array
        label_positions = np.argwhere(labels == label)
        
        # Find the maximum value for the current label
        max_val = np.max(arr[label_positions])
        
        # Store the maximum value for the current label in the output array
        max_vals[i] = max_val
    
    return max_vals
