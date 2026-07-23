# core/edge_enhancement.py
"""
Edge enhancement for leaf images
Improves contour detection and morphological analysis
"""
import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def enhance_edges(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Enhance edges in the image for better segmentation.
    
    Args:
        image: Input image (BGR)
        mask: Optional binary mask to restrict enhancement area
    
    Returns:
        Edge-enhanced image
    """
    if image is None:
        return None
    
    # Create a copy
    enhanced = image.copy()
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Detect edges using Canny
    edges = cv2.Canny(enhanced_gray, 50, 150)
    
    # Dilate edges slightly
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # If mask provided, restrict edges to mask area
    if mask is not None:
        edges = cv2.bitwise_and(edges, mask)
    
    # Create edge overlay
    edge_overlay = np.zeros_like(image)
    edge_overlay[edges > 0] = [0, 255, 0]  # Green edges
    
    # Blend with original
    enhanced = cv2.addWeighted(image, 0.7, edge_overlay, 0.3, 0)
    
    return enhanced


def amplify_leaf_edges(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Amplify leaf edges for better contour detection.
    
    Args:
        image: Input image
        mask: Optional binary mask
    
    Returns:
        Edge-amplified image
    """
    if image is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply unsharp mask for edge amplification
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    unsharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    
    # Convert back to color
    amplified = cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR)
    
    # If mask provided, combine with original
    if mask is not None:
        # Use original for non-leaf areas
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        amplified = (amplified * mask_3ch + image * (1 - mask_3ch)).astype(np.uint8)
    
    return amplified


def sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Sharpen image using a sharpening kernel.
    
    Args:
        image: Input image
        strength: Sharpening strength (0.5-2.0)
    
    Returns:
        Sharpened image
    """
    if image is None:
        return None
    
    # Create sharpening kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 4 + strength, -1],
        [0, -1, 0]
    ])
    
    # Apply kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Enhance contrast using CLAHE.
    
    Args:
        image: Input image
        clip_limit: CLAHE clip limit
    
    Returns:
        Contrast-enhanced image
    """
    if image is None:
        return None
    
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def detect_leaf_boundary(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Detect leaf boundary from mask.
    
    Args:
        image: Input image
        mask: Binary mask
    
    Returns:
        Boundary overlay image
    """
    if image is None or mask is None:
        return image
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Create boundary overlay
    overlay = image.copy()
    
    # Draw all contours
    for contour in contours:
        # Smooth contour
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Draw boundary
        cv2.drawContours(overlay, [approx], -1, (0, 255, 0), 2)
    
    # Blend with original
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    return result


def enhance_veins(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Enhance leaf vein structure.
    
    Args:
        image: Input image
        mask: Optional binary mask
    
    Returns:
        Vein-enhanced image
    """
    if image is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply Frangi filter-like enhancement
    # (Simplified: use Hessian matrix approximation)
    sigma = 1.5
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigma)
    
    # Calculate Hessian matrix components
    dx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    dxx = cv2.Sobel(dx, cv2.CV_32F, 1, 0, ksize=3)
    dyy = cv2.Sobel(dy, cv2.CV_32F, 0, 1, ksize=3)
    dxy = cv2.Sobel(dx, cv2.CV_32F, 0, 1, ksize=3)
    
    # Calculate eigenvalues
    trace = dxx + dyy
    det = dxx * dyy - dxy * dxy
    discriminant = np.sqrt(np.maximum(trace**2 / 4 - det, 0))
    lambda1 = trace/2 + discriminant
    lambda2 = trace/2 - discriminant
    
    # Vein enhancement response
    response = np.abs(lambda2) * (np.abs(lambda2) > np.abs(lambda1))
    response = np.maximum(0, response)
    
    # Normalize
    response = response / (np.max(response) + 1e-6)
    response = (response * 255).astype(np.uint8)
    
    # Apply mask
    if mask is not None:
        response = cv2.bitwise_and(response, mask)
    
    # Blend with original
    color_response = cv2.cvtColor(response, cv2.COLOR_GRAY2BGR)
    enhanced_image = cv2.addWeighted(image, 0.7, color_response, 0.3, 0)
    
    return enhanced_image