# utils/image_utils.py
"""
Image processing utilities
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def load_image(path: Union[str, Path], 
               mode: str = 'color') -> Optional[np.ndarray]:
    """
    Load an image from file.
    
    Args:
        path: Path to image file
        mode: 'color', 'grayscale', or 'unchanged'
    
    Returns:
        Image as numpy array, or None if failed
    """
    try:
        if mode == 'color':
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        elif mode == 'grayscale':
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        
        if img is None:
            logger.error(f"Failed to load image: {path}")
            return None
        
        return img
    except Exception as e:
        logger.error(f"Error loading image {path}: {e}")
        return None


def save_image(image: np.ndarray, path: Union[str, Path]) -> bool:
    """
    Save an image to file.
    
    Args:
        image: Image to save
        path: Output path
    
    Returns:
        True if successful
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(path), image)
        if not success:
            logger.error(f"Failed to save image: {path}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error saving image {path}: {e}")
        return False


def resize_image(image: np.ndarray, 
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 scale: Optional[float] = None,
                 maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize an image.
    
    Args:
        image: Input image
        width: Target width
        height: Target height
        scale: Scale factor
        maintain_aspect: Whether to maintain aspect ratio
    
    Returns:
        Resized image
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    if scale is not None:
        width = int(w * scale)
        height = int(h * scale)
    
    if width is None and height is None:
        return image
    
    if maintain_aspect:
        if width is not None and height is None:
            ratio = width / w
            height = int(h * ratio)
        elif height is not None and width is None:
            ratio = height / h
            width = int(w * ratio)
    
    return cv2.resize(image, (width, height))


def convert_color_space(image: np.ndarray, 
                        from_space: str = 'BGR',
                        to_space: str = 'RGB') -> np.ndarray:
    """
    Convert image between color spaces.
    
    Args:
        image: Input image
        from_space: Source color space
        to_space: Target color space
    
    Returns:
        Converted image
    """
    if image is None:
        return None
    
    if len(image.shape) == 2:
        # Grayscale - can't convert to color
        return image
    
    # Common conversions
    conversions = {
        ('BGR', 'RGB'): cv2.COLOR_BGR2RGB,
        ('RGB', 'BGR'): cv2.COLOR_RGB2BGR,
        ('BGR', 'HSV'): cv2.COLOR_BGR2HSV,
        ('HSV', 'BGR'): cv2.COLOR_HSV2BGR,
        ('BGR', 'LAB'): cv2.COLOR_BGR2LAB,
        ('LAB', 'BGR'): cv2.COLOR_LAB2BGR,
        ('BGR', 'RGBA'): cv2.COLOR_BGR2RGBA,
        ('RGBA', 'BGR'): cv2.COLOR_RGBA2BGR,
        ('BGR', 'GRAY'): cv2.COLOR_BGR2GRAY,
        ('RGB', 'GRAY'): cv2.COLOR_RGB2GRAY,
    }
    
    key = (from_space.upper(), to_space.upper())
    if key in conversions:
        return cv2.cvtColor(image, conversions[key])
    else:
        logger.warning(f"Unsupported color conversion: {from_space} -> {to_space}")
        return image


def add_text_overlay(image: np.ndarray, text: str, 
                     position: Tuple[int, int] = (10, 30),
                     font_scale: float = 0.7,
                     color: Tuple[int, int, int] = (255, 255, 255),
                     thickness: int = 2) -> np.ndarray:
    """
    Add text overlay to an image.
    
    Args:
        image: Input image
        text: Text to add
        position: (x, y) position
        font_scale: Font size
        color: Text color (BGR)
        thickness: Line thickness
    
    Returns:
        Image with text overlay
    """
    if image is None:
        return None
    
    result = image.copy()
    cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)
    return result


def create_thumbnail(image: np.ndarray, 
                     max_size: int = 200) -> np.ndarray:
    """
    Create a thumbnail image.
    
    Args:
        image: Input image
        max_size: Maximum dimension size
    
    Returns:
        Thumbnail image
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    scale = min(max_size / h, max_size / w, 1.0)
    
    if scale < 1.0:
        return resize_image(image, scale=scale)
    return image.copy()


def overlay_mask(image: np.ndarray, mask: np.ndarray,
                 color: Tuple[int, int, int] = (0, 255, 0),
                 alpha: float = 0.3) -> np.ndarray:
    """
    Overlay a mask on an image.
    
    Args:
        image: Input image
        mask: Binary mask
        color: Overlay color (BGR)
        alpha: Transparency of overlay
    
    Returns:
        Image with mask overlay
    """
    if image is None or mask is None:
        return image
    
    # Ensure mask matches image size
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Create colored overlay
    overlay = np.zeros_like(image)
    overlay[mask > 127] = color
    
    # Blend with original
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    return result