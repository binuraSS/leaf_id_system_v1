# core/analysis/validation.py
"""
Validation module for leaf isolation quality
"""
import cv2
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def validate_leaf_isolation(cropped_leaf: np.ndarray, mask: np.ndarray, 
                            original_image: np.ndarray) -> Tuple[float, List, bool]:
    """
    Validate the quality of leaf isolation.
    
    Args:
        cropped_leaf: Cropped leaf image
        mask: Binary mask
        original_image: Original image
    
    Returns:
        confidence: Confidence score (0-1)
        checks: List of validation check results
        passed: Boolean indicating if validation passed
    """
    checks = []
    passed = True
    confidence = 0.0
    
    # Check 1: Emptiness check
    emptiness_check = check_emptiness(cropped_leaf)
    checks.append(f"Emptiness: {emptiness_check}")
    if not emptiness_check:
        passed = False
    
    # Check 2: Coverage check
    coverage_check = check_coverage(mask, original_image)
    checks.append(f"Coverage: {coverage_check:.1%}")
    if coverage_check < 0.05:
        passed = False
    
    # Check 3: Border touch check
    border_check = check_border_touch(mask, original_image)
    checks.append(f"Border touch: {border_check}")
    if border_check:
        passed = False
    
    # Check 4: Leaf size check
    size_check = check_leaf_size(mask)
    checks.append(f"Leaf size: {size_check:.1%}")
    if size_check < 0.01:
        passed = False
    
    # Check 5: Mask quality check
    quality_check = check_mask_quality(mask)
    checks.append(f"Mask quality: {quality_check:.1%}")
    if quality_check < 0.3:
        passed = False
    
    # Calculate overall confidence
    confidence = calculate_validation_confidence(checks)
    
    return confidence, checks, passed


def check_emptiness(image: np.ndarray) -> bool:
    """
    Check if the image is not completely empty.
    
    Args:
        image: Input image
    
    Returns:
        True if image has content
    """
    if image is None:
        return False
    
    # Check if image has any non-zero pixels
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            # RGBA - check alpha or color channels
            has_color = np.any(image[:, :, :3] > 0)
            has_alpha = np.any(image[:, :, 3] > 0)
            return has_color or has_alpha
        else:
            return np.any(image > 0)
    else:
        return np.any(image > 0)


def check_coverage(mask: np.ndarray, image: np.ndarray) -> float:
    """
    Check what percentage of the image is covered by the mask.
    
    Args:
        mask: Binary mask
        image: Original image
    
    Returns:
        Coverage ratio (0-1)
    """
    if mask is None or image is None:
        return 0.0
    
    mask_area = np.sum(mask > 127)
    image_area = image.shape[0] * image.shape[1]
    
    return mask_area / image_area


def check_border_touch(mask: np.ndarray, image: np.ndarray) -> bool:
    """
    Check if the mask touches the image border.
    
    Args:
        mask: Binary mask
        image: Original image
    
    Returns:
        True if mask touches border
    """
    if mask is None:
        return True
    
    h, w = mask.shape[:2]
    
    # Check top border
    if np.any(mask[0, :] > 127):
        return True
    
    # Check bottom border
    if np.any(mask[h-1, :] > 127):
        return True
    
    # Check left border
    if np.any(mask[:, 0] > 127):
        return True
    
    # Check right border
    if np.any(mask[:, w-1] > 127):
        return True
    
    return False


def check_leaf_size(mask: np.ndarray) -> float:
    """
    Check if the leaf is a reasonable size.
    
    Args:
        mask: Binary mask
    
    Returns:
        Size ratio relative to image
    """
    if mask is None:
        return 0.0
    
    mask_area = np.sum(mask > 127)
    image_area = mask.shape[0] * mask.shape[1]
    
    return mask_area / image_area


def check_mask_quality(mask: np.ndarray) -> float:
    """
    Check the quality of the mask.
    
    Args:
        mask: Binary mask
    
    Returns:
        Quality score (0-1)
    """
    if mask is None:
        return 0.0
    
    try:
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        # Calculate solidity
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Calculate circularity
        perimeter = cv2.arcLength(largest, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Combined quality score
        quality = (solidity * 0.6 + min(circularity * 2, 1) * 0.4)
        
        return min(1.0, max(0.0, quality))
        
    except Exception as e:
        logger.error(f"Mask quality check error: {e}")
        return 0.0


def calculate_validation_confidence(checks: List[str]) -> float:
    """
    Calculate confidence score from validation checks.
    
    Args:
        checks: List of validation check results
    
    Returns:
        Confidence score (0-1)
    """
    # Parse check results
    scores = []
    
    for check in checks:
        if "Empty" in check:
            if "True" in check:
                scores.append(1.0)
            else:
                scores.append(0.0)
        elif "Coverage" in check:
            # Extract percentage
            try:
                value = float(check.split(":")[1].strip().rstrip("%")) / 100
                scores.append(min(1.0, value * 2))  # Scale up for better sensitivity
            except:
                scores.append(0.5)
        elif "Border touch" in check:
            if "False" in check:
                scores.append(1.0)
            else:
                scores.append(0.0)
        elif "Leaf size" in check:
            try:
                value = float(check.split(":")[1].strip().rstrip("%")) / 100
                scores.append(min(1.0, value * 5))  # Scale up for better sensitivity
            except:
                scores.append(0.5)
        elif "Mask quality" in check:
            try:
                value = float(check.split(":")[1].strip().rstrip("%")) / 100
                scores.append(value)
            except:
                scores.append(0.5)
    
    if not scores:
        return 0.0
    
    return sum(scores) / len(scores)


def get_validation_summary(confidence: float, checks: List[str], passed: bool) -> str:
    """
    Get a summary of validation results.
    
    Args:
        confidence: Confidence score
        checks: List of check results
        passed: Boolean indicating if validation passed
    
    Returns:
        Summary string
    """
    status = "✅ PASSED" if passed else "❌ FAILED"
    
    summary = f"Validation {status} (Confidence: {confidence:.1%})\n"
    summary += "-" * 40 + "\n"
    
    for check in checks:
        summary += f"  • {check}\n"
    
    return summary