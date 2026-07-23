# config/settings.py
"""
Application configuration settings
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ModelSettings:
    """Model paths and parameters"""
    
    # Model paths
    yolo_model_path: str = "models/yolov8n.pt"
    sam_model_path: str = "models/sam2.1_t.pt"
    species_model_path: str = "models/species_model.pkl"
    
    # Detection parameters
    yolo_confidence: float = 0.25
    yolo_confidence_fallback: float = 0.10
    
    # SAM parameters
    sam_confidence: float = 0.5
    sam_iou_threshold: float = 0.9
    
    # Image processing
    max_image_size: int = 4032
    resize_target: int = 640
    
    @classmethod
    def from_env(cls) -> "ModelSettings":
        """Load from environment variables"""
        return cls(
            yolo_model_path=os.getenv("YOLO_MODEL", "models/yolov8n.pt"),
            sam_model_path=os.getenv("SAM_MODEL", "models/sam2.1_t.pt"),
            yolo_confidence=float(os.getenv("YOLO_CONFIDENCE", "0.25")),
        )


@dataclass
class AnalysisSettings:
    """Analysis parameters"""
    
    # Color analysis
    color_mode: str = "hybrid"  # "fast", "hybrid", "detailed"
    enable_color: bool = True
    
    # Disease detection
    enable_disease: bool = True
    disease_confidence: float = 0.3
    
    # Species classification
    enable_species: bool = False
    
    # Quality scoring
    enable_quality: bool = True
    
    # Validation
    enable_validation: bool = True
    validation_strictness: float = 0.7
    
    @classmethod
    def from_env(cls) -> "AnalysisSettings":
        """Load from environment variables"""
        return cls(
            color_mode=os.getenv("COLOR_MODE", "hybrid"),
            enable_disease=os.getenv("ENABLE_DISEASE", "true").lower() == "true",
            enable_species=os.getenv("ENABLE_SPECIES", "false").lower() == "true",
        )


@dataclass
class AppSettings:
    """Application settings"""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # CORS
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Upload
    max_upload_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'])
    
    # Output
    output_dir: str = "outputs"
    upload_dir: str = "uploads"
    cache_dir: str = "cache"
    
    # Performance
    enable_cache: bool = True
    max_cache_size: int = 100
    
    @classmethod
    def from_env(cls) -> "AppSettings":
        """Load from environment variables"""
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            debug=os.getenv("DEBUG", "true").lower() == "true",
        )


@dataclass
class Settings:
    """Master settings class"""
    
    models: ModelSettings = field(default_factory=ModelSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    app: AppSettings = field(default_factory=AppSettings)
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load all settings from environment variables"""
        return cls(
            models=ModelSettings.from_env(),
            analysis=AnalysisSettings.from_env(),
            app=AppSettings.from_env()
        )


# Default settings instance
settings = Settings()