# core/analysis/species_classifier.py
"""
Species classification for leaf identification
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class SpeciesClassifier:
    """
    Classify plant species from leaf images.
    
    Uses a combination of morphological features and color signatures
    to identify leaf species.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the species classifier.
        
        Args:
            model_path: Path to trained model file
        """
        self.model_path = model_path or "species_model.pkl"
        self.model = None
        self.species_list = []
        self.is_trained = False
        
        # Load model if exists
        if os.path.exists(self.model_path):
            self.load_model(self.model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to model file
        
        Returns:
            True if model loaded successfully
        """
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data.get('model')
            self.species_list = data.get('species_list', [])
            self.is_trained = True
            
            logger.info(f"Loaded species model with {len(self.species_list)} species")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load species model: {e}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """
        Save trained model to file.
        
        Args:
            model_path: Path to save model
        
        Returns:
            True if model saved successfully
        """
        try:
            data = {
                'model': self.model,
                'species_list': self.species_list
            }
            with open(model_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved species model to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save species model: {e}")
            return False
    
    def predict(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Predict species from leaf image.
        
        Args:
            image: Input image (BGR)
            mask: Binary mask of leaf
        
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained or self.model is None:
            return {
                'error': 'Model not trained',
                'species': 'Unknown',
                'confidence': 0.0
            }
        
        try:
            # Extract features
            features = self._extract_features(image, mask)
            
            if features is None:
                return {
                    'error': 'Failed to extract features',
                    'species': 'Unknown',
                    'confidence': 0.0
                }
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba([features])[0]
                predicted_idx = np.argmax(probabilities)
                confidence = probabilities[predicted_idx]
            else:
                predicted_idx = self.model.predict([features])[0]
                confidence = 0.5
            
            species = self.species_list[predicted_idx] if predicted_idx < len(self.species_list) else 'Unknown'
            
            # Get top predictions
            top_predictions = []
            if hasattr(self.model, 'predict_proba'):
                top_indices = np.argsort(probabilities)[-3:][::-1]
                for idx in top_indices:
                    top_predictions.append({
                        'species': self.species_list[idx],
                        'confidence': float(probabilities[idx])
                    })
            
            return {
                'species': species,
                'confidence': float(confidence),
                'top_predictions': top_predictions,
                'feature_vector': features.tolist() if features is not None else []
            }
            
        except Exception as e:
            logger.error(f"Species prediction error: {e}")
            return {
                'error': str(e),
                'species': 'Unknown',
                'confidence': 0.0
            }
    
    def _extract_features(self, image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract feature vector from leaf image.
        
        Args:
            image: Input image
            mask: Binary mask
        
        Returns:
            Feature vector as numpy array
        """
        try:
            # Get leaf pixels
            leaf_pixels = image[mask > 127]
            
            if len(leaf_pixels) < 100:
                return None
            
            features = []
            
            # 1. Color features
            bgr_mean = np.mean(leaf_pixels, axis=0)
            bgr_std = np.std(leaf_pixels, axis=0)
            
            # Convert to HSV
            hsv_pixels = cv2.cvtColor(leaf_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            hsv_mean = np.mean(hsv_pixels, axis=0)
            hsv_std = np.std(hsv_pixels, axis=0)
            
            features.extend(bgr_mean)
            features.extend(bgr_std)
            features.extend(hsv_mean)
            features.extend(hsv_std)
            
            # 2. Morphological features
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                
                area = cv2.contourArea(largest)
                perimeter = cv2.arcLength(largest, True)
                
                x, y, w, h = cv2.boundingRect(largest)
                aspect_ratio = w / h if h > 0 else 0
                
                hull = cv2.convexHull(largest)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                
                features.append(area / (mask.shape[0] * mask.shape[1]))  # Normalized area
                features.append(aspect_ratio)
                features.append(solidity)
                features.append(circularity)
                features.append(len(largest) / 100)  # Contour complexity
            
            # 3. Texture features (simplified)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            leaf_grays = masked_gray[mask > 127]
            
            if len(leaf_grays) > 0:
                features.append(np.mean(leaf_grays) / 255.0)
                features.append(np.std(leaf_grays) / 255.0)
                
                # Gradient features
                grad_x = cv2.Sobel(masked_gray, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(masked_gray, cv2.CV_32F, 0, 1, ksize=3)
                magnitude = np.sqrt(grad_x**2 + grad_y**2)
                magnitude_masked = magnitude[mask > 127]
                
                if len(magnitude_masked) > 0:
                    features.append(np.mean(magnitude_masked) / 255.0)
                    features.append(np.std(magnitude_masked) / 255.0)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    def train(self, images: List[np.ndarray], masks: List[np.ndarray], 
              labels: List[str]) -> bool:
        """
        Train the species classifier.
        
        Args:
            images: List of leaf images
            masks: List of corresponding masks
            labels: List of species labels
        
        Returns:
            True if training successful
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # Extract features
            features = []
            valid_labels = []
            
            for img, mask, label in zip(images, masks, labels):
                feat = self._extract_features(img, mask)
                if feat is not None:
                    features.append(feat)
                    valid_labels.append(label)
            
            if len(features) < 2:
                logger.error("Not enough features for training")
                return False
            
            # Encode labels
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(valid_labels)
            self.species_list = label_encoder.classes_.tolist()
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model.fit(features, encoded_labels)
            
            self.is_trained = True
            logger.info(f"Trained species model with {len(self.species_list)} species")
            
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def get_species_info(self, species_name: str) -> Dict[str, Any]:
        """
        Get additional information about a species.
        
        Args:
            species_name: Name of the species
        
        Returns:
            Dictionary with species information
        """
        # This could be extended with a database of species info
        species_info = {
            'Oak': {
                'common_name': 'Oak',
                'family': 'Fagaceae',
                'notes': 'Lobed leaves, deciduous'
            },
            'Maple': {
                'common_name': 'Maple',
                'family': 'Sapindaceae',
                'notes': 'Palmate leaves, opposite arrangement'
            },
            'Rose': {
                'common_name': 'Rose',
                'family': 'Rosaceae',
                'notes': 'Compound leaves, serrated edges'
            }
        }
        
        return species_info.get(species_name, {
            'common_name': species_name,
            'family': 'Unknown',
            'notes': 'No additional information available'
        })