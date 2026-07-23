#!/bin/bash
# scripts/deploy.sh
# Deployment script for Leaf ID System

echo "🌿 Deploying Leaf ID System..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Build Docker image
echo "📦 Building Docker image..."
docker build -t leaf-id-system .

# Stop existing container if running
echo "🛑 Stopping existing container..."
docker stop leaf-id-system 2>/dev/null || true
docker rm leaf-id-system 2>/dev/null || true

# Run new container
echo "🚀 Starting new container..."
docker run -d \
    --name leaf-id-system \
    -p 8000:8000 \
    -v $(pwd)/uploads:/app/uploads \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/models:/app/models \
    --restart unless-stopped \
    leaf-id-system

echo "✅ Deployment complete!"
echo "🌐 Open http://localhost:8000 to access the app"