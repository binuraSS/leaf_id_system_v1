# tests/unit/test_analysis.py
"""
Analysis module unit tests
"""
import unittest
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.analysis.color_analyzer import HybridColorAnalyzer
from core.analysis.disease_detector import DiseaseDetector
from core.analysis.quality_scorer import calculate_quality_score, get_quality_grade


class TestAnalysis(unittest.TestCase):
    """Test analysis modules"""
    
    def setUp(self):
        """Set up test environment"""
        self.color_analyzer = HybridColorAnalyzer()
        self.disease_detector = DiseaseDetector()
        
        # Create sample image and mask
        self.sample_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.sample_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(self.sample_image, (50, 50), 30, (0, 255, 0), -1)
        cv2.circle(self.sample_mask, (50, 50), 30, 255, -1)
    
    def test_color_analysis(self):
        """Test color analysis"""
        result = self.color_analyzer.analyze(self.sample_image, self.sample_mask)
        self.assertIsNotNone(result)
        self.assertIn('mean_rgb', result)
        self.assertIn('health_indicators', result)
    
    def test_disease_detection(self):
        """Test disease detection"""
        result = self.disease_detector.detect(self.sample_image, self.sample_mask, {})
        self.assertIsNotNone(result)
        self.assertIn('summary', result)
    
    def test_quality_scoring(self):
        """Test quality scoring"""
        cropped = self.sample_image.copy()
        score, details = calculate_quality_score(cropped, self.sample_mask, self.sample_image)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        
        grade, label = get_quality_grade(score)
        self.assertIn(grade, ['A', 'B', 'C', 'D', 'F'])


if __name__ == '__main__':
    unittest.main()