# core/analysis/color_analyzer.py
"""
Hybrid color analysis for leaf health assessment
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class HybridColorAnalyzer:
    """Hybrid color analyzer for leaf health assessment"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.mode = self.config.get('COLOR_ANALYSIS', {}).get('mode', 'hybrid')
    
    def analyze(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Analyze color of leaf for health assessment.
        
        Args:
            image: Input image (BGR)
            mask: Binary mask of leaf
        
        Returns:
            Dictionary with color analysis results
        """
        if image is None or mask is None:
            return {'error': 'Missing image or mask'}
        
        # Ensure mask matches image
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Get leaf pixels only
        leaf_pixels = image[mask > 127]
        
        if len(leaf_pixels) == 0:
            return {'error': 'No leaf pixels found'}
        
        # Calculate color spaces
        bgr_mean = np.mean(leaf_pixels, axis=0)
        
        # Convert to HSV
        hsv_pixels = cv2.cvtColor(leaf_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        hsv_mean = np.mean(hsv_pixels, axis=0)
        
        # Convert to LAB
        lab_pixels = cv2.cvtColor(leaf_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        lab_mean = np.mean(lab_pixels, axis=0)
        
        # Calculate health indicators
        health_indicators = self._calculate_health_indicators(
            bgr_mean, hsv_mean, lab_mean, leaf_pixels
        )
        
        # Calculate advanced metrics
        advanced_metrics = self._calculate_advanced_metrics(leaf_pixels, hsv_pixels)
        
        # Get dominant colors
        dominant_colors = self._get_dominant_colors(leaf_pixels)
        
        return {
            'mean_rgb': bgr_mean.tolist(),
            'mean_hsv': hsv_mean.tolist(),
            'mean_lab': lab_mean.tolist(),
            'color_spaces': {
                'RGB': {'mean': bgr_mean.tolist()},
                'HSV': {'mean': hsv_mean.tolist()},
                'LAB': {'mean': lab_mean.tolist()}
            },
            'health_indicators': health_indicators,
            'advanced_metrics': advanced_metrics,
            'dominant_colors': dominant_colors,
            'dominant_hex': self._rgb_to_hex(bgr_mean)
        }
    
    def _calculate_health_indicators(self, bgr_mean, hsv_mean, lab_mean, leaf_pixels):
        """Calculate health indicators from color data"""
        # Nitrogen status based on green channel
        green_channel = bgr_mean[1]
        red_channel = bgr_mean[2]
        
        # Green/Red ratio as nitrogen proxy
        g_r_ratio = green_channel / (red_channel + 1)
        
        # Normalize to 0-100
        g_r_normalized = min(100, max(0, (g_r_ratio / 1.5) * 100))
        
        if g_r_ratio > 1.2:
            nitrogen_status = "Optimal"
            nitrogen_score = 90
        elif g_r_ratio > 0.8:
            nitrogen_status = "Normal"
            nitrogen_score = 70
        elif g_r_ratio > 0.5:
            nitrogen_status = "Deficient"
            nitrogen_score = 45
        else:
            nitrogen_status = "Severe Deficiency"
            nitrogen_score = 20
        
        # Stress level based on color variation
        bgr_std = np.std(leaf_pixels, axis=0)
        color_variance = np.mean(bgr_std)
        
        if color_variance < 20:
            stress_level = "Low"
            stress_score = 20
        elif color_variance < 40:
            stress_level = "Moderate"
            stress_score = 50
        else:
            stress_level = "High"
            stress_score = 80
        
        # Overall health score
        health_score = max(0, min(100, 
            0.5 * g_r_normalized + 
            0.3 * (100 - stress_score) + 
            0.2 * nitrogen_score
        ))
        
        return {
            'overall_health_score': health_score,
            'nitrogen_status': nitrogen_status,
            'stress_level': stress_level,
            'green_red_ratio': g_r_ratio,
            'color_variance': float(color_variance)
        }
    
    def _calculate_advanced_metrics(self, leaf_pixels, hsv_pixels):
        """Calculate advanced color metrics"""
        # Chlorophyll index (based on green hue)
        hsv_hue = hsv_pixels[:, 0]
        green_hue_indices = (hsv_hue >= 40) & (hsv_hue <= 80)
        
        if np.sum(green_hue_indices) > 0:
            chlorophyll_index = np.mean(hsv_pixels[green_hue_indices, 1]) / 255.0
        else:
            chlorophyll_index = 0.5
        
        # Carotenoid index (based on yellow-orange hues)
        carotenoid_hue_indices = (hsv_hue >= 20) & (hsv_hue < 40)
        if np.sum(carotenoid_hue_indices) > 0:
            carotenoid_index = np.mean(hsv_pixels[carotenoid_hue_indices, 1]) / 255.0
        else:
            carotenoid_index = 0.3
        
        return {
            'chlorophyll_index': float(chlorophyll_index * 100),
            'carotenoid_index': float(carotenoid_index * 100),
            'nitrogen_index': float(chlorophyll_index * 100)
        }
    
    def _get_dominant_colors(self, leaf_pixels, num_colors=5):
        """Get dominant colors from leaf pixels"""
        if len(leaf_pixels) < num_colors:
            return []
        
        # Use K-means clustering
        pixels = leaf_pixels.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        try:
            _, labels, centers = cv2.kmeans(
                pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )
            
            # Count pixels per cluster
            counts = np.bincount(labels.flatten())
            
            # Sort by count
            sorted_indices = np.argsort(counts)[::-1]
            
            dominant_colors = []
            for idx in sorted_indices:
                bgr = centers[idx].astype(np.uint8)
                rgb = bgr[::-1]  # BGR to RGB
                hex_color = self._rgb_to_hex(rgb)
                percentage = counts[idx] / len(pixels) * 100
                
                dominant_colors.append({
                    'rgb': rgb.tolist(),
                    'hex': hex_color,
                    'percentage': float(percentage)
                })
            
            return dominant_colors
            
        except Exception as e:
            logger.error(f"K-means clustering error: {e}")
            return []
    
    def _rgb_to_hex(self, rgb):
        """Convert RGB to hex color string"""
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def get_health_assessment(self, color_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get health assessment from color results"""
        if 'error' in color_results:
            return {'error': color_results['error']}
        
        indicators = color_results.get('health_indicators', {})
        
        return {
            'overall_health_score': indicators.get('overall_health_score', 0),
            'nitrogen_status': indicators.get('nitrogen_status', 'Unknown'),
            'stress_level': indicators.get('stress_level', 'Unknown'),
            'green_red_ratio': indicators.get('green_red_ratio', 0),
            'color_variance': indicators.get('color_variance', 0)
        }