# utils/validators.py
"""
Validation utilities for images and files
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_image(image: np.ndarray) -> bool:
    """
    Validate that the input is a valid image.
    
    Args:
        image: Image to validate
    
    Returns:
        True if valid
    """
    if image is None:
        logger.error("Image is None")
        return False
    
    if not isinstance(image, np.ndarray):
        logger.error(f"Image is not numpy array: {type(image)}")
        return False
    
    if len(image.shape) < 2:
        logger.error(f"Image has invalid shape: {image.shape}")
        return False
    
    if image.shape[0] == 0 or image.shape[1] == 0:
        logger.error(f"Image has zero dimension: {image.shape}")
        return False
    
    return True


def validate_mask(mask: np.ndarray, 
                  image: Optional[np.ndarray] = None) -> bool:
    """
    Validate that the input is a valid binary mask.
    
    Args:
        mask: Mask to validate
        image: Optional image to check dimensions against
    
    Returns:
        True if valid
    """
    if not validate_image(mask):
        return False
    
    # Check if mask is binary
    unique_values = np.unique(mask)
    if not all(v in [0, 255] for v in unique_values):
        if not all(v in [0, 1] for v in unique_values):
            logger.warning(f"Mask has non-binary values: {unique_values}")
            return False
    
    # Check dimensions match image
    if image is not None and mask.shape[:2] != image.shape[:2]:
        logger.error(f"Mask shape {mask.shape} does not match image {image.shape}")
        return False
    
    return True


def validate_file_extension(filename: str, 
                           allowed_extensions: list = None) -> bool:
    """
    Validate file extension against allowed list.
    
    Args:
        filename: File name or path
        allowed_extensions: List of allowed extensions
    
    Returns:
        True if extension is allowed
    """
    if allowed_extensions is None:
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        logger.error(f"Invalid file extension: {ext}. Allowed: {allowed_extensions}")
        return False
    
    return True


def validate_file_size(file_size: int, max_size: int = 50 * 1024 * 1024) -> bool:
    """
    Validate file size against maximum limit.
    
    Args:
        file_size: Size in bytes
        max_size: Maximum allowed size in bytes
    
    Returns:
        True if size is within limit
    """
    if file_size > max_size:
        logger.error(f"File too large: {file_size / (1024*1024):.1f}MB. Max: {max_size / (1024*1024):.1f}MB")
        return False
    
    return True


def validate_image_path(path: Union[str, Path]) -> bool:
    """
    Validate that an image file exists and is readable.
    
    Args:
        path: Path to image file
    
    Returns:
        True if valid
    """
    path = Path(path)
    
    if not path.exists():
        logger.error(f"File does not exist: {path}")
        return False
    
    if not validate_file_extension(path.name):
        return False
    
    # Try to read image
    img = cv2.imread(str(path))
    if img is None:
        logger.error(f"Failed to read image: {path}")
        return False
    
    return True


def validate_color_metrics(metrics: dict) -> bool:
    """
    Validate color metrics dictionary.
    
    Args:
        metrics: Color metrics dictionary
    
    Returns:
        True if valid
    """
    required_keys = ['mean_rgb', 'mean_hsv', 'mean_lab']
    
    for key in required_keys:
        if key not in metrics:
            logger.error(f"Missing required key: {key}")
            return False
        
        value = metrics[key]
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
            logger.error(f"Invalid {key}: {value}")
            return False
    
    return True


def validate_disease_results(results: dict) -> bool:
    """
    Validate disease detection results.
    
    Args:
        results: Disease results dictionary
    
    Returns:
        True if valid
    """
    required_keys = ['summary', 'metrics']
    
    for key in required_keys:
        if key not in results:
            logger.error(f"Missing required key: {key}")
            return False
    
    summary_keys = ['chlorosis_percentage', 'necrosis_percentage', 'status']
    summary = results.get('summary', {})
    
    for key in summary_keys:
        if key not in summary:
            logger.error(f"Missing summary key: {key}")
            return False
    
    return True


def validate_pipeline_result(result) -> bool:
    """
    Validate pipeline result object.
    
    Args:
        result: PipelineResult object
    
    Returns:
        True if valid
    """
    if result is None:
        logger.error("Result is None")
        return False
    
    # Check required fields
    if not hasattr(result, 'success'):
        logger.error("Result missing 'success' attribute")
        return False
    
    if not hasattr(result, 'processing_time'):
        logger.error("Result missing 'processing_time' attribute")
        return False
    
    if not hasattr(result, 'image_path'):
        logger.error("Result missing 'image_path' attribute")
        return False
    
    return True