# app/routes/batch.py
"""
Batch processing route - multiple images
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import time
from typing import List

from core.pipeline import PipelineOrchestrator

router = APIRouter(prefix="/api", tags=["Batch"])
orchestrator = PipelineOrchestrator()


@router.post("/analyze-batch")
async def analyze_batch(files: List[UploadFile] = File(...)):
    """
    Analyze multiple leaf images in batch
    
    Upload up to 10 images at once
    """
    try:
        if len(files) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 images per batch"
            )
        
        print(f"📸 Batch processing {len(files)} images")
        total_start = time.time()
        
        results = []
        for idx, file in enumerate(files):
            try:
                print(f"   Processing {idx+1}/{len(files)}: {file.filename}")
                
                # Read and decode image
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "Invalid image format"
                    })
                    continue
                
                # Convert RGBA to BGR if needed
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                
                # Run analysis
                result = orchestrator.run(img)
                
                results.append({
                    "filename": file.filename,
                    "success": result.success,
                    "health_score": result.health.overall_health_score if result.health else 0,
                    "disease_status": result.disease.status if result.disease else "Unknown",
                    "quality_grade": result.quality.grade if result.quality else "F",
                    "processing_time": result.processing_time
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "total": len(files),
            "successful": sum(1 for r in results if r.get("success", False)),
            "total_time": time.time() - total_start,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )