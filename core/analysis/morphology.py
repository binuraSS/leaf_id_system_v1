# core/analysis/morphology.py
"""
Morphology analysis for leaf shape and size metrics
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def analyze_morphology(mask: np.ndarray, pixel_to_cm: float = 0.01) -> Dict[str, Any]:
    """
    Calculate morphology metrics from leaf mask.
    
    Args:
        mask: Binary mask of leaf
        pixel_to_cm: Conversion factor from pixels to cm
    
    Returns:
        Dictionary with morphology metrics
    """
    if mask is None:
        return {'error': 'No mask provided'}
    
    try:
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'error': 'No contours found'}
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Basic measurements
        area_pixels = cv2.contourArea(largest)
        perimeter_pixels = cv2.arcLength(largest, True)
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(largest)
        
        # Convex hull
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, True)
        
        # Calculate metrics
        aspect_ratio = w / h if h > 0 else 0
        solidity = area_pixels / hull_area if hull_area > 0 else 0
        circularity = (4 * np.pi * area_pixels) / (perimeter_pixels ** 2) if perimeter_pixels > 0 else 0
        convexity = hull_perimeter / perimeter_pixels if perimeter_pixels > 0 else 0
        
        # Equivalent diameter
        equivalent_diameter = np.sqrt(4 * area_pixels / np.pi)
        
        # Minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        
        # Convert to cm if conversion factor provided
        area_cm2 = area_pixels * (pixel_to_cm ** 2)
        perimeter_cm = perimeter_pixels * pixel_to_cm
        
        return {
            'area_pixels': float(area_pixels),
            'area_cm2': float(area_cm2),
            'perimeter_pixels': float(perimeter_pixels),
            'perimeter_cm': float(perimeter_cm),
            'aspect_ratio': float(aspect_ratio),
            'solidity': float(solidity),
            'circularity': float(circularity),
            'convexity': float(convexity),
            'equivalent_diameter': float(equivalent_diameter),
            'bounding_box': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            },
            'center': {
                'x': float(cx),
                'y': float(cy)
            },
            'radius': float(radius),
            'hull_area': float(hull_area),
            'contour_points': len(largest)
        }
        
    except Exception as e:
        logger.error(f"Morphology analysis error: {e}")
        return {'error': str(e)}


def calculate_leaf_complexity(mask: np.ndarray) -> float:
    """
    Calculate leaf shape complexity using fractal dimension approach.
    
    Args:
        mask: Binary mask
    
    Returns:
        Complexity score (0-1)
    """
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        largest = max(contours, key=cv2.contourArea)
        
        # Calculate perimeter at different resolutions
        perimeters = []
        for epsilon in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]:
            approx = cv2.approxPolyDP(largest, epsilon * cv2.arcLength(largest, True), True)
            perimeters.append(cv2.arcLength(approx, True))
        
        # Calculate complexity as variation in perimeter
        if perimeters:
            mean_perimeter = np.mean(perimeters)
            std_perimeter = np.std(perimeters)
            complexity = std_perimeter / mean_perimeter if mean_perimeter > 0 else 0
            
            # Normalize to 0-1
            return min(1.0, complexity * 2)
        
        return 0.0
        
    except Exception as e:
        logger.error(f"Complexity calculation error: {e}")
        return 0.0


def extract_leaf_features(mask: np.ndarray) -> Dict[str, Any]:
    """
    Extract comprehensive leaf features.
    
    Args:
        mask: Binary mask
    
    Returns:
        Dictionary with extracted features
    """
    morphology = analyze_morphology(mask)
    
    if 'error' in morphology:
        return morphology
    
    # Additional features
    try:
        # Eccentricity of the leaf
        moments = cv2.moments(mask)
        if moments['mu20'] > 0 and moments['mu02'] > 0:
            eccentricity = np.sqrt(1 - (moments['mu02'] / moments['mu20']))
        else:
            eccentricity = 0
        
        # Extent: area / bounding box area
        bbox_area = morphology['bounding_box']['width'] * morphology['bounding_box']['height']
        extent = morphology['area_pixels'] / bbox_area if bbox_area > 0 else 0
        
        features = {
            **morphology,
            'eccentricity': float(eccentricity),
            'extent': float(extent),
            'has_holes': has_holes(mask)
        }
        
        return features
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return morphology


def has_holes(mask: np.ndarray) -> bool:
    """
    Check if mask has holes.
    
    Args:
        mask: Binary mask
    
    Returns:
        True if mask has holes
    """
    try:
        # Invert mask
        inverted = cv2.bitwise_not(mask)
        
        # Find contours
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any contour is fully inside the mask
        for contour in contours:
            if cv2.pointPolygonTest(contour, (mask.shape[1]//2, mask.shape[0]//2), False) >= 0:
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Hole detection error: {e}")
        return False