# app/main.py
"""
FastAPI application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.routes import analysis, batch, health, upload
from config.settings import settings

# Create FastAPI app
app = FastAPI(
    title="🌿 Leaf ID System API",
    description="Leaf identification and health analysis using YOLO + SAM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.app.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routes
app.include_router(health.router, tags=["Health"])
app.include_router(upload.router, tags=["Upload"])
app.include_router(analysis.router, tags=["Analysis"])
app.include_router(batch.router, tags=["Batch"])


@app.get("/")
async def root():
    """Serve the dashboard"""
    from fastapi.responses import HTMLResponse
    from pathlib import Path
    
    template_path = Path(__file__).parent / "templates" / "dashboard.html"
    if template_path.exists():
        with open(template_path, "r") as f:
            return HTMLResponse(content=f.read())
    
    return {"message": "🌿 Leaf ID System API", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Leaf ID System"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug
    )