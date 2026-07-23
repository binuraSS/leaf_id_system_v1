#!/usr/bin/env python3
# scripts/train_model.py
"""
Script to train the species classification model
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
from core.analysis.species_classifier import SpeciesClassifier
from tests.fixtures.sample_images import get_test_images


def train_species_model():
    """Train the species classification model"""
    print("🌿 Training species classification model...")
    
    # Load sample data (replace with real data)
    images = []
    masks = []
    labels = []
    
    # This is a placeholder - replace with actual dataset
    test_data = get_test_images()
    
    # Create synthetic training data
    for species in ['oak', 'maple', 'rose']:
        for _ in range(10):
            img, mask = create_sample_leaf_image()
            images.append(img)
            masks.append(mask)
            labels.append(species)
    
    # Initialize classifier
    classifier = SpeciesClassifier("species_model.pkl")
    
    # Train
    success = classifier.train(images, masks, labels)
    
    if success:
        print(f"✅ Model trained successfully with {len(classifier.species_list)} species")
        print(f"   Species: {', '.join(classifier.species_list)}")
    else:
        print("❌ Training failed")


def create_sample_leaf_image():
    """Create a sample leaf image for training"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)
    
    # Draw different leaf shapes based on species
    # This is a simplified version
    center = (100, 100)
    axes = (60, 40)
    
    # Random variation
    angle = np.random.randint(-30, 30)
    color = np.random.randint(100, 200, 3).tolist()
    
    cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    
    return img, mask


if __name__ == "__main__":
    train_species_model()