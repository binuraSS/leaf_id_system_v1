# tests/integration/test_api.py
"""
API integration tests
"""
import unittest
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient
from app.main import app


class TestAPI(unittest.TestCase):
    """Test API endpoints"""
    
    def setUp(self):
        """Set up test client"""
        self.client = TestClient(app)
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
    
    def test_analyze_with_image(self):
        """Test analyze endpoint with sample image"""
        # Create a sample image file
        import cv2
        import numpy as np
        from io import BytesIO
        
        # Create sample image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 30, (0, 255, 0), -1)
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = BytesIO(buffer.tobytes())
        
        # Upload
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        response = self.client.post("/api/analyze", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success", False))
    
    def test_upload_endpoint(self):
        """Test upload endpoint"""
        import cv2
        import numpy as np
        from io import BytesIO
        
        # Create sample image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = BytesIO(buffer.tobytes())
        
        # Upload
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        response = self.client.post("/api/upload", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success", False))
        self.assertIn("filename", data)


if __name__ == '__main__':
    unittest.main()