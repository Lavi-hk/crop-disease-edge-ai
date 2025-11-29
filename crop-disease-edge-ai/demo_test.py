#!/usr/bin/env python3
"""
Quick demo test - Run inference on test images
No hardware required - pure software testing
"""

import cv2
import numpy as np
from pathlib import Path
from src.preprocessing import ImagePreprocessor
from src.inference_engine import InferenceEngine

def demo_inference():
    """Demo inference on test images"""

    print("\n" + "="*60)
    print("🌾 CROP DISEASE DETECTION - DEMO TEST")
    print("="*60)

    # Load class labels
    with open("data/class_labels.txt", "r") as f:
        class_labels = [line.strip() for line in f.readlines()]

    print(f"\n✅ Loaded {len(class_labels)} disease classes")

    # Check if model exists
    model_path = "models/model.tflite"
    if not Path(model_path).exists():
        print(f"\n⚠️  Warning: {model_path} not found")
        print("   You need to download/train a model first")
        print("   See docs/DEPLOYMENT.md for instructions")
        return

    # Initialize
    print(f"\n📦 Loading model from {model_path}...")
    try:
        engine = InferenceEngine(model_path, class_labels)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    preprocessor = ImagePreprocessor()

    # Find test images
    test_dir = Path("data/test_images")
    if not test_dir.exists():
        print(f"\n⚠️  No test images found in {test_dir}")
        print("   Copy leaf images to data/test_images/")
        return

    test_images = list(test_dir.glob("*.jpg"))
    if not test_images:
        print(f"\n⚠️  No JPG files found in {test_dir}")
        return

    print(f"\n🖼️  Found {len(test_images)} test images")
    print("\n" + "-"*60)
    print("INFERENCE RESULTS")
    print("-"*60)

    results = []
    for i, img_path in enumerate(test_images, 1):
        try:
            # Load image
            image = preprocessor.load_image(img_path)

            # Run inference
            prediction = engine.predict(image)

            results.append(prediction)

            # Print result
            print(f"\n{i}. {img_path.name}")
            print(f"   Disease: {prediction['class_name']}")
            print(f"   Confidence: {prediction['confidence']:.2%}")
            print(f"   Top 3:")

            # Get top 3
            top_3_idx = np.argsort(prediction['all_predictions'])[-3:][::-1]
            for rank, idx in enumerate(top_3_idx, 1):
                conf = prediction['all_predictions'][idx]
                print(f"      {rank}. {class_labels[idx]}: {conf:.2%}")

        except Exception as e:
            print(f"\n❌ Error processing {img_path.name}: {e}")

    print("\n" + "-"*60)
    print(f"✅ Processed {len(results)} images successfully!")
    print("="*60 + "\n")

if __name__ == "__main__":
    demo_inference()
