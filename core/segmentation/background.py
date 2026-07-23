# core/segmentation/background.py
"""
Background subtraction and mask refinement using SAM
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def try_sam_with_fallback(image, sam_model, enhanced_img, best_box, 
                          img_w, img_h, screen_center):
    """
    Try SAM segmentation with fallback options.
    
    Args:
        image: Original image
        sam_model: SAM model instance
        enhanced_img: Edge-enhanced image
        best_box: Best bounding box from YOLO
        img_w: Image width
        img_h: Image height
        screen_center: Center of screen
    
    Returns:
        mask: Binary mask
        success: Boolean indicating success
    """
    try:
        # First attempt with SAM
        mask = generate_sam_mask(sam_model, image, best_box)
        
        if mask is not None and np.sum(mask) > 1000:
            logger.info("SAM segmentation successful")
            return mask, True
        
        # Fallback: Use Otsu thresholding
        logger.warning("SAM failed, using Otsu threshold fallback")
        mask = otsu_fallback(image)
        
        if mask is not None and np.sum(mask) > 500:
            return mask, True
        
        # Second fallback: Use whole image as mask
        logger.warning("All segmentation methods failed, using full image mask")
        mask = np.ones((img_h, img_w), dtype=np.uint8) * 255
        return mask, False
        
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        # Return full image mask as last resort
        mask = np.ones((img_h, img_w), dtype=np.uint8) * 255
        return mask, False


def generate_sam_mask(sam_model, image, box):
    """
    Generate mask using SAM with box prompt.
    
    Args:
        sam_model: SAM model
        image: Input image
        box: Bounding box [x1, y1, x2, y2]
    
    Returns:
        Binary mask
    """
    try:
        # Run SAM prediction with box prompt
        results = sam_model.predict(image, bboxes=box, verbose=False)[0]
        
        if results and len(results.masks) > 0:
            # Get the first mask
            mask = results.masks[0].data.cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            
            # Ensure mask is binary
            mask = (mask > 127).astype(np.uint8) * 255
            
            return mask
        
        return None
        
    except Exception as e:
        logger.error(f"SAM prediction error: {e}")
        return None


def otsu_fallback(image):
    """
    Otsu thresholding fallback for segmentation.
    
    Args:
        image: Input image (BGR)
    
    Returns:
        Binary mask
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Otsu threshold
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            
            # Create new mask with only the largest contour
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest], -1, 255, -1)
        
        return mask
        
    except Exception as e:
        logger.error(f"Otsu fallback error: {e}")
        return None


def clean_mask(mask, image=None):
    """
    Clean and refine the mask.
    
    Args:
        mask: Binary mask
        image: Original image (optional, for guided filtering)
    
    Returns:
        Cleaned mask
    """
    if mask is None:
        return None
    
    # Ensure binary
    if mask.dtype != np.uint8:
        mask = (mask > 127).astype(np.uint8) * 255
    
    # Apply morphological operations
    kernel = np.ones((3, 3), np.uint8)
    
    # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Dilate slightly to ensure edge coverage
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Erode back to original size
    mask = cv2.erode(mask, kernel, iterations=1)
    
    return mask


def apply_alpha_channel(image, mask):
    """
    Apply alpha channel to image based on mask.
    
    Args:
        image: Input image (BGR or RGB)
        mask: Binary mask
    
    Returns:
        RGBA image with transparency
    """
    if len(image.shape) == 2:
        # Grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Create RGBA image
    if image.shape[2] == 3:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        rgba = image.copy()
    
    # Apply mask to alpha channel
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    rgba[:, :, 3] = np.where(mask > 127, 255, 0)
    
    return rgba