# core/segmentation/mask_processor.py
"""
Mask processing and refinement utilities
Combines mask extraction and refinement functionality
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def process_mask(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    Process and clean a binary mask.
    
    Args:
        mask: Binary mask
        min_area: Minimum area to keep
    
    Returns:
        Processed mask
    """
    if mask is None:
        return None
    
    # Ensure binary
    if mask.dtype != np.uint8:
        mask = (mask > 127).astype(np.uint8) * 255
    
    # Remove small components
    mask = remove_small_components(mask, min_area)
    
    # Fill holes
    mask = fill_holes(mask)
    
    # Smooth boundaries
    mask = smooth_boundaries(mask)
    
    return mask


def refine_mask(mask: np.ndarray, image: Optional[np.ndarray] = None,
                iterations: int = 3) -> np.ndarray:
    """
    Refine mask boundaries for better segmentation.
    
    Args:
        mask: Binary mask
        image: Optional original image for guided refinement
        iterations: Number of refinement iterations
    
    Returns:
        Refined mask
    """
    if mask is None:
        return None
    
    # Ensure binary
    if mask.dtype != np.uint8:
        mask = (mask > 127).astype(np.uint8) * 255
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    for _ in range(iterations):
        # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Open to remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # If image provided, use edge guidance
    if image is not None:
        mask = edge_guided_refinement(mask, image)
    
    return mask


def remove_small_components(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    Remove small connected components from mask.
    
    Args:
        mask: Binary mask
        min_area: Minimum area to keep
    
    Returns:
        Cleaned mask
    """
    if mask is None:
        return None
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    # Create new mask
    result = np.zeros_like(mask)
    
    # Keep components with area > min_area
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = 255
    
    return result


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in the mask.
    
    Args:
        mask: Binary mask
    
    Returns:
        Mask with holes filled
    """
    if mask is None:
        return None
    
    # Invert mask
    inverted = cv2.bitwise_not(mask)
    
    # Find contours
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create result
    result = mask.copy()
    
    # Fill holes
    for contour in contours:
        # Check if contour is inside mask
        if cv2.pointPolygonTest(contour, (mask.shape[1]//2, mask.shape[0]//2), False) >= 0:
            cv2.drawContours(result, [contour], -1, 255, -1)
    
    return result


def smooth_boundaries(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Smooth mask boundaries.
    
    Args:
        mask: Binary mask
        kernel_size: Size of smoothing kernel
    
    Returns:
        Smoothed mask
    """
    if mask is None:
        return None
    
    # Apply Gaussian blur to smooth boundaries
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
    
    # Threshold back to binary
    _, result = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    return result.astype(np.uint8)


def edge_guided_refinement(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    Refine mask using edge information from image.
    
    Args:
        mask: Binary mask
        image: Original image
    
    Returns:
        Refined mask
    """
    if mask is None or image is None:
        return mask
    
    try:
        # Detect edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        # Refine each contour
        result = mask.copy()
        
        for contour in contours:
            # Smooth contour
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Draw refined contour
            cv2.drawContours(result, [approx], -1, 255, -1)
        
        return result
        
    except Exception as e:
        logger.error(f"Edge-guided refinement error: {e}")
        return mask


def extract_leaf_region(image: np.ndarray, mask: np.ndarray,
                        padding: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract leaf region from image using mask.
    
    Args:
        image: Input image
        mask: Binary mask
        padding: Padding around leaf
    
    Returns:
        Tuple of (cropped_image, cropped_mask)
    """
    if image is None or mask is None:
        return None, None
    
    # Find bounding box
    coords = np.where(mask > 127)
    if len(coords[0]) == 0:
        return image, mask
    
    y_min = max(0, np.min(coords[0]) - padding)
    y_max = min(mask.shape[0], np.max(coords[0]) + padding)
    x_min = max(0, np.min(coords[1]) - padding)
    x_max = min(mask.shape[1], np.max(coords[1]) + padding)
    
    # Crop
    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    
    return cropped_image, cropped_mask


def combine_masks(masks: List[np.ndarray], method: str = 'union') -> np.ndarray:
    """
    Combine multiple masks.
    
    Args:
        masks: List of masks
        method: 'union', 'intersection', or 'majority'
    
    Returns:
        Combined mask
    """
    if not masks:
        return None
    
    # Ensure all masks are binary
    binary_masks = []
    for m in masks:
        if m is not None:
            binary_masks.append((m > 127).astype(np.uint8) * 255)
    
    if not binary_masks:
        return None
    
    if method == 'union':
        # OR operation
        result = binary_masks[0]
        for m in binary_masks[1:]:
            result = cv2.bitwise_or(result, m)
            
    elif method == 'intersection':
        # AND operation
        result = binary_masks[0]
        for m in binary_masks[1:]:
            result = cv2.bitwise_and(result, m)
            
    elif method == 'majority':
        # Majority voting
        stack = np.stack(binary_masks, axis=2)
        result = (np.sum(stack > 127, axis=2) >= len(binary_masks) // 2).astype(np.uint8) * 255
        
    else:
        raise ValueError(f"Unknown combine method: {method}")
    
    return result


def normalize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize mask to target size.
    
    Args:
        mask: Binary mask
        target_size: (height, width)
    
    Returns:
        Resized mask
    """
    if mask is None:
        return None
    
    if mask.shape[:2] == target_size:
        return mask
    
    return cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)