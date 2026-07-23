"""
Leaf Isolation Module - Extracts leaf from background
Fixes: Position preservation and black background issues
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_leaf_bounding_box(mask):
    """
    Get bounding box of the leaf mask.
    
    Args:
        mask: Binary mask (0 or 255)
    
    Returns:
        x, y, w, h: Bounding box coordinates
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0, 0, mask.shape[1], mask.shape[0]
    
    # Get bounding box of largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return x, y, w, h


def isolate_leaf(img, mask, mode='preserve', padding=20, background_color=(0, 0, 0, 0)):
    """
    Isolate leaf from background with position preservation.
    
    Args:
        img: Original image
        mask: Binary mask
        mode: 'preserve', 'crop', or 'padding'
            - 'preserve': Keep original position (RECOMMENDED for analysis)
            - 'crop': Crop to leaf bounding box (original behavior)
            - 'padding': Crop with padding around leaf
        padding: Padding size when mode='padding'
        background_color: RGBA background color (default: transparent)
    
    Returns:
        Isolated leaf image with alpha channel (RGBA)
    """
    # Ensure mask is binary
    if mask.dtype != np.uint8:
        mask = (mask > 127).astype(np.uint8) * 255
    
    # Ensure mask matches image size
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        logger.debug(f"Resized mask to match image: {img.shape}")
    
    if mode == 'preserve':
        """
        PRESERVE MODE - Keep leaf in original position
        This fixes both the black background AND shifting issues
        """
        # Create result with same size as original
        result = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        
        # Fill background with specified color (default: black)
        if background_color[:3] != (0, 0, 0):
            result[:] = background_color[:3]
        
        # Copy only leaf pixels to their original positions
        result[mask > 127] = img[mask > 127]
        
        # Create alpha channel (transparency)
        alpha = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        alpha[mask > 127] = 255
        
        # Combine RGB and Alpha
        result_with_alpha = np.dstack([result, alpha])
        
        logger.debug(f"Preserved leaf at original position in {img.shape} image")
        return result_with_alpha
    
    elif mode == 'padding':
        """
        PADDING MODE - Crop with padding around leaf
        Leaf is cropped but with some context
        """
        x, y, w, h = get_leaf_bounding_box(mask)
        
        # Add padding
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)
        
        # Crop image and mask
        cropped = img[y_start:y_end, x_start:x_end]
        cropped_mask = mask[y_start:y_end, x_start:x_end]
        
        # Apply mask to cropped image
        result = np.zeros_like(cropped)
        result[cropped_mask > 127] = cropped[cropped_mask > 127]
        
        # Alpha channel
        alpha = np.zeros((cropped.shape[0], cropped.shape[1], 1), dtype=np.uint8)
        alpha[cropped_mask > 127] = 255
        result_with_alpha = np.dstack([result, alpha])
        
        logger.debug(f"Cropped leaf with {padding}px padding")
        return result_with_alpha
    
    else:  # 'crop' - original behavior (SHIFTS POSITION)
        """
        CROP MODE - Original behavior
        Leaf is cropped to bounding box (position changes)
        """
        x, y, w, h = get_leaf_bounding_box(mask)
        
        # Crop image and mask
        cropped = img[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
        
        # Apply mask
        result = np.zeros_like(cropped)
        result[cropped_mask > 127] = cropped[cropped_mask > 127]
        
        # Alpha channel
        alpha = np.zeros((h, w, 1), dtype=np.uint8)
        alpha[cropped_mask > 127] = 255
        result_with_alpha = np.dstack([result, alpha])
        
        logger.debug(f"Cropped leaf to {w}x{h} (position shifted)")
        return result_with_alpha


def isolate_leaf_for_analysis(img, mask):
    """
    Specialized function for color analysis.
    Preserves original position for accurate analysis.
    Returns leaf pixels at their original positions.
    """
    # Ensure same size
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    
    # Extract leaf pixels (position preserved)
    leaf_pixels = img[mask > 127]
    
    # Also return positions if needed
    positions = np.where(mask > 127)
    
    return {
        'pixels': leaf_pixels,
        'positions': positions,
        'shape': img.shape
    }


def get_isolated_with_transparent_background(img, mask):
    """
    Get isolated leaf with transparent background.
    Preserves original position.
    """
    # Create RGBA image
    if len(img.shape) == 3 and img.shape[2] == 3:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    else:
        rgba = np.dstack([img, np.ones_like(img[:, :, 0]) * 255])
    
    # Set alpha channel based on mask
    rgba[:, :, 3] = np.where(mask > 127, 255, 0)
    
    return rgba


def create_mask_overlay(img, mask, color=(0, 255, 0), alpha=0.3):
    """
    Create an overlay showing where the mask is.
    Useful for debugging position issues.
    """
    overlay = img.copy()
    
    # Create colored mask
    colored_mask = np.zeros_like(img)
    colored_mask[mask > 127] = color
    
    # Blend
    result = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
    
    # Draw bounding box
    x, y, w, h = get_leaf_bounding_box(mask)
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
    # Add position info
    info = f"Mask: ({x}, {y}) to ({x+w}, {y+h})"
    cv2.putText(result, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (255, 255, 255), 2)
    
    return result


def compare_isolated_modes(img, mask):
    """
    Compare all isolation modes side by side.
    Useful for debugging and visualization.
    """
    # Get all modes
    preserve = isolate_leaf(img, mask, mode='preserve')
    padding = isolate_leaf(img, mask, mode='padding', padding=30)
    crop = isolate_leaf(img, mask, mode='crop')
    
    # Resize for display
    h, w = crop.shape[:2]
    preserve_resized = cv2.resize(preserve, (w, h))
    padding_resized = cv2.resize(padding, (w, h))
    
    # Stack horizontally
    comparison = np.hstack([crop, padding_resized, preserve_resized])
    
    # Add labels
    cv2.putText(comparison, "CROP (shifted)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, "PADDING", (w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, "PRESERVE (fixed)", (2*w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return comparison