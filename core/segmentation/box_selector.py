# core/segmentation/box_selector.py
"""
Bounding box selection for leaf detection
"""
import numpy as np
import cv2


def select_best_box(boxes, screen_center, image_shape):
    """
    Select the best bounding box from YOLO detections.
    
    Args:
        boxes: YOLO boxes object
        screen_center: Center of the image (x, y)
        image_shape: Shape of the image (h, w, c)
    
    Returns:
        Best box coordinates [x1, y1, x2, y2] or None
    """
    if boxes is None or len(boxes) == 0:
        return None
    
    best_box = None
    best_score = -1
    
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].cpu().numpy()
        
        # Calculate box center
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        
        # Calculate distance from screen center
        distance = np.sqrt((box_center_x - screen_center[0])**2 + 
                          (box_center_y - screen_center[1])**2)
        
        # Calculate area of box
        area = (x2 - x1) * (y2 - y1)
        
        # Calculate score: preference for larger boxes closer to center
        # Higher confidence and larger area get better scores
        image_area = image_shape[0] * image_shape[1]
        area_ratio = area / image_area
        
        # Combined score: confidence * area_ratio * (1 - distance/normalized_distance)
        normalized_distance = distance / np.sqrt(image_area)
        score = conf * area_ratio * (1 - normalized_distance * 0.5)
        
        if score > best_score:
            best_score = score
            best_box = [x1, y1, x2, y2]
    
    return best_box


def get_tight_bbox(mask, padding=10):
    """
    Get tight bounding box around mask with padding.
    
    Args:
        mask: Binary mask
        padding: Padding pixels to add
    
    Returns:
        x, y, w, h: Bounding box coordinates
    """
    # Find non-zero pixels
    coords = np.where(mask > 127)
    if len(coords[0]) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    
    y_min = max(0, np.min(coords[0]) - padding)
    y_max = min(mask.shape[0], np.max(coords[0]) + padding)
    x_min = max(0, np.min(coords[1]) - padding)
    x_max = min(mask.shape[1], np.max(coords[1]) + padding)
    
    return x_min, y_min, x_max - x_min, y_max - y_min


def crop_to_bbox(image, mask, bbox):
    """
    Crop image and mask to bounding box.
    
    Args:
        image: Image to crop
        mask: Mask to crop
        bbox: (x, y, w, h) bounding box
    
    Returns:
        cropped_image, cropped_mask
    """
    x, y, w, h = bbox
    
    # Crop image
    cropped_image = image[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    return cropped_image, cropped_mask