# core/analysis/quality_scorer.py
"""
Quality scoring for leaf isolation
"""
import cv2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_quality_score(cropped_leaf: np.ndarray, mask: np.ndarray, 
                            original_image: np.ndarray) -> Tuple[float, dict]:
    """
    Calculate quality score for leaf isolation.
    
    Args:
        cropped_leaf: Cropped leaf image
        mask: Binary mask
        original_image: Original image
    
    Returns:
        quality_score: Score from 0-1
        details: Dictionary with individual metric scores
    """
    if cropped_leaf is None or mask is None or original_image is None:
        return 0.0, {'error': 'Missing inputs'}
    
    details = {}
    
    # Metric 1: Mask solidity
    solidity_score = calculate_solidity(mask)
    details['solidity'] = solidity_score
    
    # Metric 2: Mask smoothness
    smoothness_score = calculate_smoothness(mask)
    details['smoothness'] = smoothness_score
    
    # Metric 3: Aspect ratio
    aspect_score = calculate_aspect_ratio(mask)
    details['aspect_ratio'] = aspect_score
    
    # Metric 4: Coverage
    coverage_score = calculate_coverage(mask, original_image)
    details['coverage'] = coverage_score
    
    # Metric 5: Edge quality
    edge_score = calculate_edge_quality(cropped_leaf, mask)
    details['edge_quality'] = edge_score
    
    # Combined score (weighted average)
    weights = {
        'solidity': 0.25,
        'smoothness': 0.20,
        'aspect_ratio': 0.15,
        'coverage': 0.20,
        'edge_quality': 0.20
    }
    
    total_score = 0
    total_weight = 0
    
    for key, value in details.items():
        if key in weights:
            total_score += value * weights[key]
            total_weight += weights[key]
    
    final_score = total_score / total_weight if total_weight > 0 else 0
    
    return final_score, details


def calculate_solidity(mask: np.ndarray) -> float:
    """
    Calculate mask solidity (area / convex hull area).
    
    Args:
        mask: Binary mask
    
    Returns:
        Solidity score (0-1)
    """
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        
        if hull_area > 0:
            return min(1.0, area / hull_area)
        else:
            return 0.0
            
    except Exception as e:
        logger.error(f"Solidity calculation error: {e}")
        return 0.0


def calculate_smoothness(mask: np.ndarray) -> float:
    """
    Calculate mask smoothness based on edge variation.
    
    Args:
        mask: Binary mask
    
    Returns:
        Smoothness score (0-1)
    """
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        largest = max(contours, key=cv2.contourArea)
        
        # Calculate perimeter
        perimeter = cv2.arcLength(largest, True)
        
        # Approximate with fewer points
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(largest, epsilon, True)
        
        # Calculate smoothness as ratio of original to approximated points
        original_points = len(largest)
        approx_points = len(approx)
        
        if original_points > 0:
            smoothness = 1.0 - (approx_points / original_points)
            return max(0.0, min(1.0, smoothness))
        else:
            return 0.0
            
    except Exception as e:
        logger.error(f"Smoothness calculation error: {e}")
        return 0.0


def calculate_aspect_ratio(mask: np.ndarray) -> float:
    """
    Calculate aspect ratio score (ideal is around 1.5-3 for leaves).
    
    Args:
        mask: Binary mask
    
    Returns:
        Aspect ratio score (0-1)
    """
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        if h == 0:
            return 0.0
        
        aspect_ratio = w / h
        
        # Ideal ratio for leaves is between 1.5 and 3.0
        if 1.5 <= aspect_ratio <= 3.0:
            return 1.0
        elif 1.0 <= aspect_ratio <= 4.0:
            return 0.7
        elif 0.5 <= aspect_ratio <= 5.0:
            return 0.4
        else:
            return 0.2
            
    except Exception as e:
        logger.error(f"Aspect ratio calculation error: {e}")
        return 0.0


def calculate_coverage(mask: np.ndarray, original_image: np.ndarray) -> float:
    """
    Calculate coverage score.
    
    Args:
        mask: Binary mask
        original_image: Original image
    
    Returns:
        Coverage score (0-1)
    """
    if mask is None or original_image is None:
        return 0.0
    
    mask_area = np.sum(mask > 127)
    image_area = original_image.shape[0] * original_image.shape[1]
    
    coverage = mask_area / image_area
    
    # Ideal coverage is 10-50% of image
    if 0.10 <= coverage <= 0.50:
        return 1.0
    elif 0.05 <= coverage <= 0.70:
        return 0.7
    elif 0.02 <= coverage <= 0.80:
        return 0.4
    else:
        return 0.2


def calculate_edge_quality(cropped_leaf: np.ndarray, mask: np.ndarray) -> float:
    """
    Calculate edge quality based on edge sharpness.
    
    Args:
        cropped_leaf: Cropped leaf image
        mask: Binary mask
    
    Returns:
        Edge quality score (0-1)
    """
    try:
        # Convert to grayscale
        if len(cropped_leaf.shape) == 3:
            gray = cv2.cvtColor(cropped_leaf, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped_leaf
        
        # Apply mask
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Detect edges
        edges = cv2.Canny(masked_gray, 50, 150)
        
        # Calculate edge density
        edge_pixels = np.sum(edges > 0)
        mask_pixels = np.sum(mask > 127)
        
        if mask_pixels == 0:
            return 0.0
        
        edge_density = edge_pixels / mask_pixels
        
        # Ideal edge density is around 0.02-0.08
        if 0.02 <= edge_density <= 0.08:
            return 1.0
        elif 0.01 <= edge_density <= 0.12:
            return 0.7
        elif 0.005 <= edge_density <= 0.15:
            return 0.4
        else:
            return 0.2
            
    except Exception as e:
        logger.error(f"Edge quality calculation error: {e}")
        return 0.0


def get_quality_grade(score: float) -> Tuple[str, str]:
    """
    Convert quality score to grade and label.
    
    Args:
        score: Quality score (0-1)
    
    Returns:
        grade: Letter grade (A, B, C, D, F)
        label: Descriptive label
    """
    if score >= 0.85:
        return "A", "Excellent"
    elif score >= 0.70:
        return "B", "Good"
    elif score >= 0.55:
        return "C", "Fair"
    elif score >= 0.40:
        return "D", "Poor"
    else:
        return "F", "Unacceptable"