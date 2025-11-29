"""
Unit tests for inference engine
"""
import unittest
import numpy as np
import cv2
from pathlib import Path

class TestInferenceEngine(unittest.TestCase):
    """Test inference engine"""

    def test_imports(self):
        """Test module imports"""
        from src.preprocessing import ImagePreprocessor
        from src.inference_engine import InferenceEngine
        self.assertIsNotNone(ImagePreprocessor)
        self.assertIsNotNone(InferenceEngine)

    def test_preprocessor(self):
        """Test image preprocessor"""
        from src.preprocessing import ImagePreprocessor

        preprocessor = ImagePreprocessor(target_size=(224, 224))

        # Create dummy image
        dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

        # Test preprocessing
        processed = preprocessor.preprocess_image(dummy_image)
        self.assertEqual(processed.shape, (224, 224, 3))
        self.assertTrue(processed.min() >= 0)
        self.assertTrue(processed.max() <= 1)

if __name__ == '__main__':
    unittest.main()
