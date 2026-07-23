# tests/fixtures/sample_images.py
"""
Sample images for testing
"""
import cv2
import numpy as np
from pathlib import Path


def create_sample_leaf_image(size=(256, 256)):
    """
    Create a synthetic leaf image for testing.
    
    Args:
        size: (width, height) of image
    
    Returns:
        Tuple of (image, mask)
    """
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw a leaf-like shape
    center = (w // 2, h // 2)
    axes = (w // 3, h // 4)
    angle = 0
    
    # Draw ellipse for leaf
    cv2.ellipse(img, center, axes, angle, 0, 360, (0, 180, 0), -1)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    
    # Add some texture
    noise = np.random.randint(0, 20, (h, w, 3), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.9, noise, 0.1, 0)
    
    # Add vein-like lines
    cv2.line(img, (w//2, h//4), (w//2, 3*h//4), (0, 100, 0), 2)
    cv2.line(img, (w//4, h//2), (3*w//4, h//2), (0, 100, 0), 2)
    
    return img, mask


def create_sample_diseased_leaf(size=(256, 256)):
    """
    Create a synthetic diseased leaf for testing.
    
    Args:
        size: (width, height) of image
    
    Returns:
        Tuple of (image, mask)
    """
    img, mask = create_sample_leaf_image(size)
    w, h = size
    
    # Add chlorosis (yellow patches)
    for _ in range(3):
        x = np.random.randint(w//4, 3*w//4)
        y = np.random.randint(h//4, 3*h//4)
        r = np.random.randint(10, 30)
        cv2.circle(img, (x, y), r, (0, 200, 200), -1)
    
    # Add necrosis (brown patches)
    for _ in range(2):
        x = np.random.randint(w//4, 3*w//4)
        y = np.random.randint(h//4, 3*h//4)
        r = np.random.randint(5, 15)
        cv2.circle(img, (x, y), r, (0, 50, 100), -1)
    
    # Add spots
    for _ in range(5):
        x = np.random.randint(w//4, 3*w//4)
        y = np.random.randint(h//4, 3*h//4)
        r = np.random.randint(2, 5)
        cv2.circle(img, (x, y), r, (0, 0, 0), -1)
    
    return img, mask


def get_test_images():
    """
    Get sample test images.
    
    Returns:
        Dictionary of test images
    """
    healthy_img, healthy_mask = create_sample_leaf_image()
    diseased_img, diseased_mask = create_sample_diseased_leaf()
    
    return {
        'healthy': {
            'image': healthy_img,
            'mask': healthy_mask
        },
        'diseased': {
            'image': diseased_img,
            'mask': diseased_mask
        }
    }