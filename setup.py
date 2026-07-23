# setup.py
"""
Setup file for leaf identification system
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="leaf-id-system",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Leaf Identification and Health Analysis System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/leaf-id-system",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "ultralytics>=8.0.0",
        "pillow>=10.0.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "scikit-learn>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "leaf-id=app.main:main",
        ],
    },
)