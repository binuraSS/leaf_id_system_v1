# app/routes/health.py
"""
Health check endpoint
"""
from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """Check if the service is running"""
    return {
        "status": "healthy",
        "service": "Leaf ID System",
        "version": "1.0.0"
    }