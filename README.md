# ONNX Model Conversion & Object Detection Toolkit

**Purpose:**
Convert trained models to ONNX format for optimized inference, and perform object detection using ONNX Runtime.

## Prerequisites
```bash
python3.8+
pip
```

## Install
```bash
git clone https://github.com/rickaha/ComputerVision.git
cd ComputerVision
pip install -r requirements.txt
```

## Usage

### 1. Model Conversion (`export.py`)
Convert a trained model to ONNX format for optimized inference.

### 2. Object Detection (`detection.py`)
Perform object detection using ONNX models.

**requirements.txt**
```bash
numpy>=1.20.0
onnx>=1.12.0
onnxruntime>=1.12.0
torch>=1.10.0  # If converting PyTorch models
opencv-python>=4.5.0
pyyaml>=6.0
```
