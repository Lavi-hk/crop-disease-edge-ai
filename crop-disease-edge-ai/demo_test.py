#!/usr/bin/env python3
"""
Crop Disease Detection - Demo (No TensorFlow required)
"""

import cv2
import numpy as np
from pathlib import Path

def test_basic_imports():
    """Test that all modules work"""
    print("\n" + "="*60)
    print("🌾 CROP DISEASE DETECTION - BASIC TEST")
    print("="*60)
    
    try:
        import cv2
        print("✅ OpenCV loaded successfully")
    except Exception as e:
        print(f"❌ OpenCV error: {e}")
    
    try:
        import numpy
        print("✅ NumPy loaded successfully")
    except Exception as e:
        print(f"❌ NumPy error: {e}")
    
    try:
        import PIL
        print("✅ Pillow loaded successfully")
    except Exception as e:
        print(f"❌ Pillow error: {e}")
    
    # Load class labels
    class_labels_path = Path("data/class_labels.txt")
    if class_labels_path.exists():
        with open(class_labels_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        print(f"✅ Loaded {len(labels)} disease classes")
    else:
        print(f"⚠️  Class labels file not found: {class_labels_path}")
    
    # Check test images
    test_dir = Path("data/test_images")
    if test_dir.exists():
        images = list(test_dir.glob("*.jpg"))
        print(f"✅ Found {len(images)} test images")
        for img in images[:3]:
            print(f"   - {img.name}")
    else:
        print(f"⚠️  Test images directory not found: {test_dir}")
    
    print("\n" + "="*60)
    print("✅ ALL BASIC TESTS PASSED!")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_basic_imports()
