# core/pipeline.py
"""
Main pipeline orchestrator - coordinates all analysis steps
"""
import time
import cv2
import numpy as np
from typing import Optional, Union
from pathlib import Path

from core.segmentation.isolator import isolate_leaf
from core.segmentation.box_selector import select_best_box
from core.segmentation.background import try_sam_with_fallback
from core.analysis.validation import validate_leaf_isolation
from core.analysis.quality_scorer import calculate_quality_score, get_quality_grade
from core.analysis.color_analyzer import HybridColorAnalyzer
from core.analysis.disease_detector import DiseaseDetector

from models.result import (
    PipelineResult, HealthMetrics, DiseaseMetrics, QualityMetrics,
    MorphologyMetrics, ColorMetrics
)
from config.settings import settings

from ultralytics import YOLO, SAM


class PipelineOrchestrator:
    """Main orchestrator for the leaf analysis pipeline"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        # Load models
        print("Loading YOLO model...")
        self.yolo_model = YOLO(settings.models.yolo_model_path)
        print("Loading SAM model...")
        self.sam_model = SAM(settings.models.sam_model_path)
        print("Models loaded successfully!")
        
        # Initialize analyzers
        self.color_analyzer = HybridColorAnalyzer()
        self.disease_detector = DiseaseDetector()
    
    def run(self, image: Union[str, Path, np.ndarray]) -> PipelineResult:
        """
        Run the full analysis pipeline on a leaf image
        
        Args:
            image: Path to image or numpy array (BGR)
            
        Returns:
            PipelineResult with all analysis data
        """
        start_time = time.time()
        phase_metrics = {}
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            image_path = str(image)
            if img is None:
                return self._error_result(
                    "Failed to load image", 
                    time.time() - start_time, 
                    str(image)
                )
        else:
            img = image
            image_path = "uploaded_image"
        
        try:
            # Step 1: Generate mask
            print("🔍 Step 1: Generating segmentation mask...")
            mask = self._generate_mask(img)
            if mask is None:
                return self._error_result(
                    "Failed to generate leaf mask",
                    time.time() - start_time,
                    image_path
                )
            
            # Step 2: Isolate leaf
            print("🌿 Step 2: Isolating leaf...")
            cropped_leaf = isolate_leaf(img, mask, mode='preserve')
            if cropped_leaf is None:
                return self._error_result(
                    "Failed to isolate leaf",
                    time.time() - start_time,
                    image_path,
                    mask=mask
                )
            
            # Step 3: Validate
            print("✅ Step 3: Validating isolation...")
            if cropped_leaf.shape[2] == 4:
                cropped_leaf_bgr = cv2.cvtColor(cropped_leaf, cv2.COLOR_BGRA2BGR)
            else:
                cropped_leaf_bgr = cropped_leaf
            
            confidence, _, passed = validate_leaf_isolation(
                cropped_leaf_bgr, mask, img
            )
            
            if not passed and np.sum(mask) < 5000:
                return self._error_result(
                    "Validation failed - leaf too small or poorly segmented",
                    time.time() - start_time,
                    image_path,
                    mask=mask,
                    cropped_leaf=cropped_leaf
                )
            
            # Step 4: Quality scoring
            print("📊 Step 4: Calculating quality metrics...")
            quality_score, _ = calculate_quality_score(cropped_leaf_bgr, mask, img)
            grade, grade_label = get_quality_grade(quality_score)
            quality_metrics = QualityMetrics(
                score=quality_score,
                grade=grade,
                label=grade_label
            )
            
            # Step 5: Color analysis
            print("🎨 Step 5: Color analysis...")
            health_metrics, color_metrics = self._analyze_color(
                cropped_leaf_bgr, mask
            )
            
            # Step 6: Disease detection
            print("🦠 Step 6: Disease detection...")
            disease_metrics = self._analyze_disease(
                cropped_leaf_bgr, mask
            )
            
            # Step 7: Morphology analysis
            print("📏 Step 7: Morphology analysis...")
            morphology_metrics = self._analyze_morphology(mask)
            
            # Build result
            return PipelineResult(
                success=True,
                processing_time=time.time() - start_time,
                image_path=image_path,
                mask=mask,
                cropped_leaf=cropped_leaf,
                original_image=img,
                health=health_metrics,
                disease=disease_metrics,
                quality=quality_metrics,
                morphology=morphology_metrics,
                color_metrics=color_metrics,
                phase_metrics=phase_metrics
            )
            
        except Exception as e:
            import traceback
            print(f"❌ Pipeline error: {e}")
            traceback.print_exc()
            return self._error_result(
                f"Analysis error: {str(e)}",
                time.time() - start_time,
                image_path
            )
    
    def _generate_mask(self, img: np.ndarray) -> np.ndarray:
        """Generate leaf mask using YOLO + SAM"""
        import cv2
        import numpy as np
        
        img_h, img_w = img.shape[:2]
        screen_center = np.array([img_w / 2, img_h / 2])
        
        # Try different confidence thresholds
        for conf in [0.25, 0.15, 0.10]:
            yolo_results = self.yolo_model.predict(img, conf=conf, verbose=False)[0]
            if yolo_results.boxes and len(yolo_results.boxes) > 0:
                best_box = select_best_box(
                    yolo_results.boxes, screen_center, img.shape
                )
                if best_box:
                    break
        else:
            best_box = [0, 0, img_w, img_h]
        
        # Get mask from SAM
        mask, _ = try_sam_with_fallback(
            img, self.sam_model, img, best_box, img_w, img_h, screen_center
        )
        
        if mask is not None and mask.shape[:2] != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h))
        
        return mask
    
    def _analyze_color(self, image: np.ndarray, mask: np.ndarray):
        """Run color analysis"""
        try:
            color_results = self.color_analyzer.analyze(image, mask)
            if color_results:
                indicators = color_results.get('health_indicators', {})
                rgb_vals = color_results.get('mean_rgb', [0, 0, 0])
                
                if len(rgb_vals) >= 3:
                    r, g, b = int(rgb_vals[0]), int(rgb_vals[1]), int(rgb_vals[2])
                    primary_hex = f"#{r:02x}{g:02x}{b:02x}"
                else:
                    primary_hex = "#00aa00"
                
                health = HealthMetrics(
                    overall_health_score=indicators.get('overall_health_score', 75.0),
                    nitrogen_status=indicators.get('nitrogen_status', 'Normal'),
                    stress_level=indicators.get('stress_level', 'Low'),
                    primary_hex=primary_hex,
                    chlorophyll_index=indicators.get('chlorophyll_index', 0.0),
                    carotenoid_index=indicators.get('carotenoid_index', 0.0)
                )
                
                color_metrics = ColorMetrics(
                    mean_rgb=tuple(rgb_vals) if len(rgb_vals) >= 3 else (0, 0, 0),
                    dominant_colors=color_results.get('dominant_colors', [])
                )
                return health, color_metrics
        except Exception as e:
            print(f"Color analysis error: {e}")
        
        return HealthMetrics(), ColorMetrics()
    
    def _analyze_disease(self, image: np.ndarray, mask: np.ndarray):
        """Run disease detection"""
        try:
            disease_results = self.disease_detector.detect(image, mask, {})
            if disease_results:
                summary = disease_results.get('summary', {})
                return DiseaseMetrics(
                    chlorosis_percentage=summary.get('chlorosis_percentage', 0.0),
                    necrosis_percentage=summary.get('necrosis_percentage', 0.0),
                    status=summary.get('status', 'Healthy'),
                    severity=summary.get('severity', 'None'),
                    spots_count=summary.get('spots_count', 0),
                    recommendations=disease_results.get('recommendations', [])
                )
        except Exception as e:
            print(f"Disease detection error: {e}")
        
        return DiseaseMetrics()
    
    def _analyze_morphology(self, mask: np.ndarray):
        """Calculate morphology metrics"""
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return MorphologyMetrics()
            
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Circularity: 4πA / P²
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            # Solidity: Area / Convex Hull Area
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            return MorphologyMetrics(
                area_pixels=area,
                perimeter_pixels=perimeter,
                aspect_ratio=aspect_ratio,
                circularity=circularity,
                solidity=solidity,
                bounding_box=(x, y, w, h)
            )
        except Exception as e:
            print(f"Morphology analysis error: {e}")
        
        return MorphologyMetrics()
    
    def _error_result(self, error_msg: str, processing_time: float, 
                     image_path: str, **kwargs) -> PipelineResult:
        """Create an error result"""
        return PipelineResult(
            success=False,
            processing_time=processing_time,
            image_path=image_path,
            validation_summary=error_msg,
            **kwargs
        )