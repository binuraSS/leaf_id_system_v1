# Makefile for Leaf ID System

.PHONY: help install run test clean docker-build docker-run deploy download-models

help:
	@echo "Available commands:"
	@echo "  make install         - Install dependencies"
	@echo "  make run             - Run the application"
	@echo "  make test            - Run tests"
	@echo "  make clean           - Clean temporary files"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run Docker container"
	@echo "  make deploy          - Deploy the application"
	@echo "  make download-models - Download ML models"

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	rm -rf htmlcov 2>/dev/null || true

docker-build:
	docker build -t leaf-id-system .

docker-run:
	docker run -d --name leaf-id-system -p 8000:8000 leaf-id-system

deploy:
	bash scripts/deploy.sh

download-models:
	python scripts/download_models.py

train-model:
	python scripts/train_model.py