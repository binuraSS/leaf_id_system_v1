# core/analysis/texture.py
"""
Texture analysis for leaf surface characteristics
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def analyze_texture(image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """
    Analyze texture of leaf surface.
    
    Args:
        image: Input image (BGR)
        mask: Binary mask
    
    Returns:
        Dictionary with texture metrics
    """
    if image is None or mask is None:
        return {'error': 'Missing image or mask'}
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply mask
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # GLCM features (simplified using local binary patterns)
        lbp_features = calculate_lbp_features(masked_gray, mask)
        
        # Statistical texture features
        stat_features = calculate_statistical_texture(masked_gray, mask)
        
        # Gabor features
        gabor_features = calculate_gabor_features(masked_gray, mask)
        
        return {
            'lbp': lbp_features,
            'statistical': stat_features,
            'gabor': gabor_features,
            'roughness': calculate_roughness(masked_gray, mask),
            'contrast': calculate_local_contrast(masked_gray, mask)
        }
        
    except Exception as e:
        logger.error(f"Texture analysis error: {e}")
        return {'error': str(e)}


def calculate_lbp_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate Local Binary Pattern features.
    
    Args:
        image: Grayscale image
        mask: Binary mask
    
    Returns:
        LBP feature dictionary
    """
    try:
        # Simple LBP implementation
        h, w = image.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = image[i, j]
                code = 0
                
                # 8-neighborhood
                neighbors = [
                    image[i-1, j-1] > center,
                    image[i-1, j] > center,
                    image[i-1, j+1] > center,
                    image[i, j+1] > center,
                    image[i+1, j+1] > center,
                    image[i+1, j] > center,
                    image[i+1, j-1] > center,
                    image[i, j-1] > center
                ]
                
                for k, val in enumerate(neighbors):
                    if val:
                        code |= (1 << k)
                
                lbp[i-1, j-1] = code
        
        # Apply mask to LBP
        lbp_masked = lbp.copy()
        lbp_masked[mask[1:-1, 1:-1] == 0] = 0
        
        # Calculate histogram
        hist, _ = np.histogram(lbp_masked, bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist = hist / (np.sum(hist) + 1e-6)
        
        # Get summary statistics
        return {
            'mean': float(np.mean(hist)),
            'std': float(np.std(hist)),
            'entropy': float(-np.sum(hist * np.log2(hist + 1e-6))),
            'energy': float(np.sum(hist ** 2)),
            'uniformity': float(np.sum(hist[hist > 0] ** 2))
        }
        
    except Exception as e:
        logger.error(f"LBP calculation error: {e}")
        return {}


def calculate_statistical_texture(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistical texture features.
    
    Args:
        image: Grayscale image
        mask: Binary mask
    
    Returns:
        Statistical features dictionary
    """
    try:
        # Get pixel values from leaf area
        pixels = image[mask > 127]
        
        if len(pixels) == 0:
            return {}
        
        # Calculate statistics
        mean = np.mean(pixels)
        std = np.std(pixels)
        var = np.var(pixels)
        
        # Skewness
        skewness = np.mean(((pixels - mean) / (std + 1e-6)) ** 3)
        
        # Kurtosis
        kurtosis = np.mean(((pixels - mean) / (std + 1e-6)) ** 4) - 3
        
        return {
            'mean': float(mean),
            'std': float(std),
            'variance': float(var),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'range': float(np.max(pixels) - np.min(pixels)),
            'median': float(np.median(pixels))
        }
        
    except Exception as e:
        logger.error(f"Statistical texture error: {e}")
        return {}


def calculate_gabor_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate Gabor filter features.
    
    Args:
        image: Grayscale image
        mask: Binary mask
    
    Returns:
        Gabor features dictionary
    """
    try:
        features = {}
        
        # Different orientations and scales
        orientations = [0, 45, 90, 135]
        scales = [2, 4, 8]
        
        for orientation in orientations:
            for scale in scales:
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    (21, 21), sigma=scale, theta=np.radians(orientation),
                    lambd=10, gamma=0.5, psi=0
                )
                
                # Apply filter
                filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
                
                # Apply mask
                filtered_masked = filtered[mask > 127]
                
                if len(filtered_masked) > 0:
                    key = f'gabor_{orientation}_{scale}'
                    features[key] = float(np.mean(filtered_masked))
        
        return features
        
    except Exception as e:
        logger.error(f"Gabor features error: {e}")
        return {}


def calculate_roughness(image: np.ndarray, mask: np.ndarray) -> float:
    """
    Calculate surface roughness.
    
    Args:
        image: Grayscale image
        mask: Binary mask
    
    Returns:
        Roughness score (0-1)
    """
    try:
        # Apply mask
        masked = cv2.bitwise_and(image, image, mask=mask)
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(masked, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(masked, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude_masked = magnitude[mask > 127]
        
        if len(magnitude_masked) == 0:
            return 0.0
        
        # Normalize roughness
        roughness = np.mean(magnitude_masked) / 255.0
        
        return min(1.0, roughness)
        
    except Exception as e:
        logger.error(f"Roughness calculation error: {e}")
        return 0.0


def calculate_local_contrast(image: np.ndarray, mask: np.ndarray) -> float:
    """
    Calculate local contrast of leaf surface.
    
    Args:
        image: Grayscale image
        mask: Binary mask
    
    Returns:
        Contrast score (0-1)
    """
    try:
        # Apply mask
        masked = cv2.bitwise_and(image, image, mask=mask)
        
        # Calculate local standard deviation
        kernel_size = 15
        mean = cv2.GaussianBlur(masked, (kernel_size, kernel_size), 0)
        mean_sq = cv2.GaussianBlur(masked**2, (kernel_size, kernel_size), 0)
        std = np.sqrt(mean_sq - mean**2)
        
        # Get std from leaf area
        std_masked = std[mask > 127]
        
        if len(std_masked) == 0:
            return 0.0
        
        # Normalize
        contrast = np.mean(std_masked) / 128.0
        
        return min(1.0, contrast)
        
    except Exception as e:
        logger.error(f"Contrast calculation error: {e}")
        return 0.0