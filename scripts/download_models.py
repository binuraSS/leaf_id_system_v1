#!/usr/bin/env python3
# scripts/download_models.py
"""
Script to download required models
"""
import os
import sys
from pathlib import Path
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url, dest_path):
    """Download a file with progress"""
    try:
        logger.info(f"Downloading {os.path.basename(dest_path)}...")
        
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, int(downloaded * 100 / total_size))
            if percent % 10 == 0:
                logger.info(f"  Progress: {percent}%")
        
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        logger.info(f"✅ Downloaded to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def download_models():
    """Download all required models"""
    
    # Create models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Model URLs (official Ultralytics)
    models = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        "sam2.1_t.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/sam2.1_t.pt",
    }
    
    logger.info("🌿 Downloading models...")
    
    for model_name, url in models.items():
        model_path = models_dir / model_name
        if model_path.exists():
            logger.info(f"✅ {model_name} already exists")
            continue
        
        download_file(url, str(model_path))
    
    logger.info("✅ All models downloaded successfully!")


if __name__ == "__main__":
    download_models()