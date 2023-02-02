import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


__all__ = ['imread','imshow']
def imread(filename):
    """Read an image file and return the image data as a NumPy array.

    Args:
        filename: The name of the image file to read.

    Returns:
        The image data as a NumPy array.
    """
    # Open the image file
    with Image.open(filename) as image:
        # Convert the image to a NumPy array
        image_data = np.array(image)
    return image_data



def imshow(image_data):
    """Display an image represented by a NumPy array.

    Args:
        image_data: The image data as a NumPy array.
    """
    # Check if the image is grayscale or color
    if image_data.ndim == 2:
        # Grayscale image
        plt.imshow(image_data, cmap='gray')
    elif image_data.ndim == 3:
        # Color image
        plt.imshow(image_data)
    else:
        raise ValueError("Invalid image data shape")

    # Show the plot
    plt.show()