# tests/unit/test_pipeline.py
"""
Pipeline unit tests
"""
import unittest
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.pipeline import PipelineOrchestrator
from models.result import PipelineResult


class TestPipeline(unittest.TestCase):
    """Test pipeline functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.pipeline = PipelineOrchestrator()
        self.sample_image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add a simple leaf-like shape
        cv2.circle(self.sample_image, (50, 50), 30, (0, 255, 0), -1)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.yolo_model)
        self.assertIsNotNone(self.pipeline.sam_model)
    
    def test_mask_generation(self):
        """Test mask generation"""
        mask = self.pipeline._generate_mask(self.sample_image)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape[:2], self.sample_image.shape[:2])
    
    def test_analysis(self):
        """Test full analysis pipeline"""
        result = self.pipeline.run(self.sample_image)
        self.assertIsInstance(result, PipelineResult)
        self.assertTrue(result.success)
    
    def test_error_handling(self):
        """Test error handling with invalid input"""
        result = self.pipeline.run(None)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.validation_summary)


if __name__ == '__main__':
    unittest.main()