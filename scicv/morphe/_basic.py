import numpy as np 
import cv2

__all__ = ['dilation','erosion','black_hat','top_hat']

def dilation(image, selem):
    """Perform dilation on a 2-D image using a given structuring element.
    
    # Define the shape of the structuring element
    selem = np.array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]], dtype=np.uint8)
    """
    result = np.zeros_like(image)
    for i in range(image.shape[0] - selem.shape[0] + 1):
        for j in range(image.shape[1] - selem.shape[1] + 1):
            sub_image = image[i:i+selem.shape[0], j:j+selem.shape[1]]
            result[i, j] = np.max(sub_image + selem)
    return result

def erosion(image, selem):
    """Perform erosion on a 2-D image using a given structuring element.
    # Define the shape of the structuring element
    selem = np.array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]], dtype=np.uint8)
    
    """
    result = np.zeros_like(image)
    for i in range(image.shape[0] - selem.shape[0] + 1):
        for j in range(image.shape[1] - selem.shape[1] + 1):
            sub_image = image[i:i+selem.shape[0], j:j+selem.shape[1]]
            result[i, j] = np.min(sub_image * selem)
    return result


def black_hat(image, selem):
    """Perform black-hat transform on a 2-D image using a given structuring element."""
    erosion_result = erosion(image, selem)
    dilation_result = dilation(erosion_result, selem)
    result = image - dilation_result
    return result

def top_hat(image, selem):
    """Perform top-hat transform on a 2-D image using a given structuring element."""
    dilation_result = dilation(image, selem)
    erosion_result = erosion(dilation_result, selem)
    result = dilation_result - erosion_result
    return result

def imnoise(img):
    # Generate random noise with a Gaussian distribution
    noise = np.random.randn(*img.shape) * 20
    # Add the noise to the image
    noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy_img
        
        

"""  
# Load an image
img = cv2.imread("image.jpg")

# Generate random noise with a Gaussian distribution
noise = np.random.randn(*img.shape) * 20

# Add the noise to the image
noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)

# Save the noisy image
cv2.imwrite("noisy_image.jpg", noisy_img)

"""