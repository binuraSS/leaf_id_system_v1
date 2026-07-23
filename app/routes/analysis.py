# app/routes/analysis.py
"""
Analysis route - single image analysis
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import traceback
import time

from core.pipeline import PipelineOrchestrator
from config.settings import settings

router = APIRouter(prefix="/api", tags=["Analysis"])
orchestrator = PipelineOrchestrator()


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze a single leaf image
    
    Upload an image of a leaf to get:
    - Health score
    - Disease detection
    - Quality assessment
    - Morphology metrics
    - Color analysis
    """
    try:
        print(f"📸 Analyzing: {file.filename}")
        start_time = time.time()
        
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid image format"}
            )
        
        # Convert RGBA to BGR if needed
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Run analysis
        result = orchestrator.run(img)
        
        if not result.success:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": result.validation_summary}
            )
        
        # Build response
        response = result.to_dict()
        
        # Add images as base64
        if result.cropped_leaf is not None:
            if len(result.cropped_leaf.shape) == 3 and result.cropped_leaf.shape[2] == 4:
                leaf_img = cv2.cvtColor(result.cropped_leaf, cv2.COLOR_BGRA2BGR)
            else:
                leaf_img = result.cropped_leaf
            _, buffer = cv2.imencode('.jpg', leaf_img)
            response["leaf_image"] = base64.b64encode(buffer).decode()
        
        if result.mask is not None:
            _, buffer = cv2.imencode('.jpg', result.mask)
            response["mask"] = base64.b64encode(buffer).decode()
        
        response["processing_time"] = result.processing_time
        
        print(f"✅ Analysis complete in {time.time() - start_time:.2f}s")
        return response
        
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )