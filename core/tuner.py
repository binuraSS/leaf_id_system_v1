# core/tuner.py
"""
Parameter tuning and optimization for the pipeline
"""
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from itertools import product

logger = logging.getLogger(__name__)


class ParameterTuner:
    """
    Tune pipeline parameters for optimal performance.
    """
    
    def __init__(self, pipeline):
        """
        Initialize tuner with pipeline instance.
        
        Args:
            pipeline: Pipeline orchestrator instance
        """
        self.pipeline = pipeline
        self.best_params = {}
        self.best_score = -1
    
    def tune_yolo_confidence(self, images: List[np.ndarray], 
                             conf_thresholds: List[float] = [0.1, 0.15, 0.2, 0.25, 0.3]) -> Dict[str, Any]:
        """
        Find optimal YOLO confidence threshold.
        
        Args:
            images: List of test images
            conf_thresholds: Confidence thresholds to try
        
        Returns:
            Dictionary with optimal parameters
        """
        results = {}
        
        for conf in conf_thresholds:
            scores = []
            for img in images:
                # Run with this confidence
                mask = self.pipeline._generate_mask_with_conf(img, conf)
                if mask is not None:
                    score = self._evaluate_mask(mask)
                    scores.append(score)
            
            avg_score = np.mean(scores) if scores else 0
            results[conf] = avg_score
        
        # Find best
        best_conf = max(results, key=results.get)
        
        return {
            'optimal_confidence': best_conf,
            'scores': results
        }
    
    def tune_sam_params(self, images: List[np.ndarray],
                        iou_thresholds: List[float] = [0.7, 0.8, 0.9],
                        score_thresholds: List[float] = [0.3, 0.4, 0.5]) -> Dict[str, Any]:
        """
        Tune SAM parameters.
        
        Args:
            images: List of test images
            iou_thresholds: IOU thresholds to try
            score_thresholds: Score thresholds to try
        
        Returns:
            Dictionary with optimal parameters
        """
        results = {}
        
        for iou, score in product(iou_thresholds, score_thresholds):
            scores = []
            for img in images:
                mask = self.pipeline._generate_mask_with_sam_params(img, iou, score)
                if mask is not None:
                    score_val = self._evaluate_mask(mask)
                    scores.append(score_val)
            
            avg_score = np.mean(scores) if scores else 0
            results[(iou, score)] = avg_score
        
        # Find best
        best_params = max(results, key=results.get)
        
        return {
            'optimal_iou': best_params[0],
            'optimal_score': best_params[1],
            'scores': results
        }
    
    def _generate_mask_with_conf(self, img: np.ndarray, conf: float) -> np.ndarray:
        """Generate mask with specific confidence threshold."""
        try:
            # This is a simplified version - adapt to your pipeline
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            results = model.predict(img, conf=conf, verbose=False)[0]
            
            if results.masks is not None and len(results.masks) > 0:
                mask = results.masks[0].data.cpu().numpy()
                return (mask > 0.5).astype(np.uint8) * 255
            return None
        except Exception as e:
            logger.error(f"Error generating mask: {e}")
            return None
    
    def _generate_mask_with_sam_params(self, img: np.ndarray, 
                                        iou: float, score: float) -> np.ndarray:
        """Generate mask with specific SAM parameters."""
        # This is a placeholder - implement based on your SAM usage
        pass
    
    def _evaluate_mask(self, mask: np.ndarray) -> float:
        """
        Evaluate mask quality.
        
        Args:
            mask: Binary mask
        
        Returns:
            Quality score (0-1)
        """
        if mask is None or np.sum(mask) < 100:
            return 0.0
        
        try:
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
            
            largest = max(contours, key=cv2.contourArea)
            
            # Calculate metrics
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            
            # Solidity
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Circularity
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Coverage
            coverage = area / (mask.shape[0] * mask.shape[1])
            
            # Combined score
            score = (solidity * 0.4 + min(circularity, 1) * 0.3 + min(coverage * 5, 1) * 0.3)
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Mask evaluation error: {e}")
            return 0.0
    
    def get_optimal_params(self) -> Dict[str, Any]:
        """Get the optimal parameters found."""
        return self.best_params