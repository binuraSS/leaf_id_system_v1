# services/analysis.py
"""
Analysis service - Main service layer
"""
import time
import cv2
import numpy as np
from typing import Union, Optional, Dict, Any
from pathlib import Path
import logging

from core.pipeline import PipelineOrchestrator
from models.result import PipelineResult
from models.config import Config
from utils.logger import get_logger
from utils.image_utils import load_image, validate_image

logger = get_logger(__name__)


class AnalysisService:
    """
    Analysis service for leaf identification and health assessment.
    
    This service wraps the pipeline and provides a clean interface
    for the web application.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the analysis service.
        
        Args:
            config: Application configuration
        """
        self.config = config or Config()
        self.pipeline = PipelineOrchestrator()
        self.cache = {}  # Simple cache for results
        self.cache_size = 100
        
        logger.info("AnalysisService initialized")
    
    def analyze_image(self, image: Union[str, Path, np.ndarray],
                      cache_key: Optional[str] = None) -> PipelineResult:
        """
        Analyze a leaf image.
        
        Args:
            image: Image path, file path, or numpy array
            cache_key: Optional cache key for result caching
        
        Returns:
            PipelineResult with analysis data
        """
        start_time = time.time()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = load_image(image)
            if img is None:
                return self._error_result("Failed to load image")
            image_path = str(image)
        else:
            img = image
            image_path = "uploaded_image"
        
        # Validate image
        if not validate_image(img):
            return self._error_result("Invalid image")
        
        # Check cache
        if cache_key and cache_key in self.cache:
            logger.info(f"Cache hit for {cache_key}")
            return self.cache[cache_key]
        
        # Run pipeline
        try:
            logger.info("Running analysis pipeline...")
            result = self.pipeline.run(img)
            result.image_path = image_path
            
            # Cache result
            if cache_key and len(self.cache) < self.cache_size:
                self.cache[cache_key] = result
            
            logger.info(f"Analysis complete in {time.time() - start_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return self._error_result(f"Analysis failed: {str(e)}")
    
    def analyze_batch(self, images: list) -> list:
        """
        Analyze multiple images in batch.
        
        Args:
            images: List of images (paths or arrays)
        
        Returns:
            List of PipelineResult objects
        """
        results = []
        
        for i, image in enumerate(images):
            logger.info(f"Processing batch item {i+1}/{len(images)}")
            result = self.analyze_image(image)
            results.append(result)
        
        return results
    
    def get_health_summary(self, result: PipelineResult) -> Dict[str, Any]:
        """
        Get a health summary from the result.
        
        Args:
            result: PipelineResult object
        
        Returns:
            Health summary dictionary
        """
        if not result.success:
            return {'error': 'Analysis failed'}
        
        summary = {}
        
        # Extract health metrics
        if result.health:
            summary['health_score'] = result.health.overall_health_score
            summary['nitrogen_status'] = result.health.nitrogen_status
            summary['stress_level'] = result.health.stress_level
            summary['primary_color'] = result.health.primary_hex
        
        # Extract disease metrics
        if result.disease:
            summary['disease_status'] = result.disease.status
            summary['chlorosis'] = result.disease.chlorosis_percentage
            summary['necrosis'] = result.disease.necrosis_percentage
            summary['spots'] = result.disease.spots_count
            summary['recommendations'] = result.disease.recommendations
        
        # Extract quality metrics
        if result.quality:
            summary['quality_grade'] = result.quality.grade
            summary['quality_score'] = result.quality.score
        
        # Extract morphology metrics
        if result.morphology:
            summary['area_pixels'] = result.morphology.area_pixels
            summary['aspect_ratio'] = result.morphology.aspect_ratio
            summary['circularity'] = result.morphology.circularity
        
        return summary
    
    def clear_cache(self):
        """Clear the result cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def _error_result(self, error_msg: str) -> PipelineResult:
        """Create an error result."""
        from models.result import PipelineResult
        return PipelineResult(
            success=False,
            processing_time=0.0,
            image_path="",
            validation_summary=error_msg
        )