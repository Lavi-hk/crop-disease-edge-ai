# Deployment Guide

## No Hardware Setup Required

This guide covers **pure software deployment** without hardware models.

## Option 1: Google Colab (Recommended)

### Step 1: Setup
```python
# Create new notebook in Google Colab
# Run this cell:

!git clone https://github.com/yourusername/crop-disease-edge-ai
%cd crop-disease-edge-ai
!pip install -r requirements.txt
```

### Step 2: Test
```python
import sys
sys.path.insert(0, '/content/crop-disease-edge-ai')

from src.preprocessing import ImagePreprocessor
from src.inference_engine import InferenceEngine

print("✅ All modules loaded successfully!")
```

### Step 3: Run Demo
```python
exec(open('demo_test.py').read())
```

## Option 2: Local Python Environment

### Step 1: Install
```bash
git clone https://github.com/yourusername/crop-disease-edge-ai
cd crop-disease-edge-ai
pip install -r requirements.txt
```

### Step 2: Test
```bash
python demo_test.py
```

## Option 3: Docker Container

### Step 1: Build
```bash
docker build -t crop-disease-ai .
```

### Step 2: Run
```bash
docker run -it crop-disease-ai python demo_test.py
```

## Using the Model

### Python API

```python
from src.inference_engine import InferenceEngine
from src.preprocessing import ImagePreprocessor
import cv2

# Load model
engine = InferenceEngine('models/model.tflite', class_labels_file='data/class_labels.txt')

# Load image
image = cv2.imread('leaf_image.jpg')

# Predict
result = engine.predict(image)
print(f"Disease: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Processing

```python
from pathlib import Path

image_folder = Path('data/test_images')
results = []

for img_path in image_folder.glob('*.jpg'):
    image = cv2.imread(str(img_path))
    prediction = engine.predict(image)
    results.append({
        'image': img_path.name,
        'disease': prediction['class_name'],
        'confidence': prediction['confidence']
    })

# Results ready for analysis
```

## Jupyter Notebook

See `notebooks/` folder for interactive demos:
- `demo.ipynb` - Quick start guide
- `analysis.ipynb` - Performance analysis

## Troubleshooting

**Issue**: Model file not found
- **Solution**: Download model from Edge Impulse or train locally

**Issue**: Module import errors
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: Out of memory
- **Solution**: Process images one at a time instead of batches
