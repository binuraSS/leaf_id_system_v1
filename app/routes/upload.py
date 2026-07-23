# app/routes/upload.py
"""
File upload handling
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from datetime import datetime
from pathlib import Path

from config.settings import settings

router = APIRouter(prefix="/api", tags=["Upload"])

# Create upload directory
UPLOAD_DIR = Path(settings.app.upload_dir)
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file for analysis
    
    Returns the file path and metadata
    """
    try:
        # Validate file type
        ext = Path(file.filename).suffix.lower()
        if ext not in settings.app.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {settings.app.allowed_extensions}"
            )
        
        # Check file size
        contents = await file.read()
        file_size = len(contents)
        if file_size > settings.app.max_upload_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max: {settings.app.max_upload_size / (1024*1024):.1f}MB"
            )
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = UPLOAD_DIR / filename
        
        # Save file
        with open(filepath, "wb") as f:
            f.write(contents)
        
        return {
            "success": True,
            "filename": filename,
            "filepath": str(filepath),
            "size_bytes": file_size,
            "size_mb": file_size / (1024 * 1024)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )