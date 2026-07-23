# core/analysis/disease_detector.py
"""
Disease detection module for leaf analysis
Identifies chlorosis, necrosis, and other anomalies
"""
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DiseaseDetector:
    """Detect diseases and anomalies in leaf images"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.min_spot_area = self.config.get('min_spot_area', 50)
        self.chlorosis_threshold = self.config.get('chlorosis_threshold', 30)
        self.necrosis_threshold = self.config.get('necrosis_threshold', 40)
    
    def detect(self, image: np.ndarray, mask: np.ndarray, 
               color_ctx: Dict) -> Dict[str, Any]:
        """
        Detect diseases in leaf image.
        
        Args:
            image: Input image (BGR)
            mask: Binary mask of leaf
            color_ctx: Color analysis context
        
        Returns:
            Dictionary with disease detection results
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
        
        # Detect chlorosis (yellowing)
        chlorosis_mask, chlorosis_pixels = self._detect_chlorosis(image, mask)
        
        # Detect necrosis (browning)
        necrosis_mask, necrosis_pixels = self._detect_necrosis(image, mask)
        
        # Detect spots/lesions
        spots, spots_count = self._detect_spots(image, mask)
        
        # Calculate metrics
        total_leaf_area = np.sum(mask > 127)
        chlorosis_percentage = (chlorosis_pixels / total_leaf_area * 100) if total_leaf_area > 0 else 0
        necrosis_percentage = (necrosis_pixels / total_leaf_area * 100) if total_leaf_area > 0 else 0
        
        # Determine overall status
        status, severity = self._determine_status(
            chlorosis_percentage, necrosis_percentage, spots_count
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            chlorosis_percentage, necrosis_percentage, spots_count, status
        )
        
        return {
            'summary': {
                'chlorosis_percentage': float(chlorosis_percentage),
                'necrosis_percentage': float(necrosis_percentage),
                'spots_count': spots_count,
                'status': status,
                'severity': severity
            },
            'metrics': {
                'chlorosis_pixels': int(chlorosis_pixels),
                'necrosis_pixels': int(necrosis_pixels),
                'total_leaf_area': int(total_leaf_area),
                'spot_areas': [float(area) for area in spots]
            },
            'recommendations': recommendations,
            'masks': {
                'chlorosis': chlorosis_mask,
                'necrosis': necrosis_mask
            }
        }
    
    def _detect_chlorosis(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Detect chlorotic (yellow) regions on leaf"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Yellow color range in HSV
        lower_yellow1 = np.array([15, 30, 50])
        upper_yellow1 = np.array([30, 255, 255])
        lower_yellow2 = np.array([30, 30, 50])
        upper_yellow2 = np.array([45, 255, 255])
        
        # Create yellow masks
        yellow_mask1 = cv2.inRange(hsv, lower_yellow1, upper_yellow1)
        yellow_mask2 = cv2.inRange(hsv, lower_yellow2, upper_yellow2)
        yellow_mask = cv2.bitwise_or(yellow_mask1, yellow_mask2)
        
        # Apply leaf mask
        chlorosis_mask = cv2.bitwise_and(yellow_mask, mask)
        
        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        chlorosis_mask = cv2.morphologyEx(chlorosis_mask, cv2.MORPH_OPEN, kernel)
        chlorosis_mask = cv2.morphologyEx(chlorosis_mask, cv2.MORPH_CLOSE, kernel)
        
        # Count pixels
        pixels = np.sum(chlorosis_mask > 127)
        
        return chlorosis_mask, pixels
    
    def _detect_necrosis(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Detect necrotic (brown/dark) regions on leaf"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Brown/dark color range in HSV
        lower_brown = np.array([0, 30, 30])
        upper_brown = np.array([30, 255, 180])
        
        # Create brown mask
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Also detect dark regions in RGB
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Combine masks
        necrosis_mask = cv2.bitwise_or(brown_mask, dark_mask)
        
        # Apply leaf mask
        necrosis_mask = cv2.bitwise_and(necrosis_mask, mask)
        
        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        necrosis_mask = cv2.morphologyEx(necrosis_mask, cv2.MORPH_OPEN, kernel)
        necrosis_mask = cv2.morphologyEx(necrosis_mask, cv2.MORPH_CLOSE, kernel)
        
        # Count pixels
        pixels = np.sum(necrosis_mask > 127)
        
        return necrosis_mask, pixels
    
    def _detect_spots(self, image: np.ndarray, mask: np.ndarray) -> Tuple[List[float], int]:
        """Detect spots/lesions on leaf"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply mask
        masked_gray = cv2.bitwise_and(gray, mask)
        
        # Use adaptive threshold to find spots
        spots_mask = cv2.adaptiveThreshold(
            masked_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        spots_mask = cv2.morphologyEx(spots_mask, cv2.MORPH_OPEN, kernel)
        spots_mask = cv2.morphologyEx(spots_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(spots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        spot_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_spot_area < area < 5000:  # Filter out too small or too large
                spot_areas.append(area)
        
        return spot_areas, len(spot_areas)
    
    def _determine_status(self, chlorosis: float, necrosis: float, spots: int) -> Tuple[str, str]:
        """Determine overall disease status and severity"""
        # Combined severity score
        severity_score = (chlorosis * 0.4 + necrosis * 0.4 + min(spots * 2, 30))
        
        if severity_score < 10:
            return "Healthy", "None"
        elif severity_score < 25:
            return "Minor Issues", "Low"
        elif severity_score < 45:
            return "Diseased", "Moderate"
        elif severity_score < 70:
            return "Severely Diseased", "High"
        else:
            return "Critical", "Severe"
    
    def _generate_recommendations(self, chlorosis: float, necrosis: float, 
                                   spots: int, status: str) -> List[str]:
        """Generate recommendations based on disease detection"""
        recommendations = []
        
        if status == "Healthy":
            recommendations.append("Plant appears healthy. Continue regular care.")
            return recommendations
        
        # Chlorosis recommendations
        if chlorosis > 30:
            recommendations.append("⚠️ Chlorosis detected - check for nutrient deficiencies")
            recommendations.append("🌱 Consider nitrogen or iron supplementation")
        
        if chlorosis > 50:
            recommendations.append("🔬 Test soil pH - may be affecting nutrient uptake")
        
        # Necrosis recommendations
        if necrosis > 20:
            recommendations.append("⚠️ Necrosis detected - check for fungal infections")
            recommendations.append("🍄 Consider fungicide treatment")
        
        if necrosis > 50:
            recommendations.append("⚠️ Significant tissue damage - consult a plant specialist")
        
        # Spot recommendations
        if spots > 5:
            recommendations.append(f"🔍 {spots} spots detected - monitor for disease spread")
        
        if spots > 15:
            recommendations.append("⚠️ High spot count - isolate from other plants")
            recommendations.append("🧪 Consider laboratory testing for precise diagnosis")
        
        # General recommendations
        if status in ["Severely Diseased", "Critical"]:
            recommendations.append("🚨 Immediate action required - consult a plant pathologist")
            recommendations.append("📸 Document symptoms for professional consultation")
        
        # Add preventive recommendations
        recommendations.append("🌿 Maintain optimal watering and light conditions")
        recommendations.append("🧹 Remove affected leaves to prevent spread")
        
        return recommendations