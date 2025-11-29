"""
Image Preprocessing Module
"""
import cv2
import numpy as np
from pathlib import Path

class ImagePreprocessor:
    """Image preprocessing utilities"""

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def preprocess_image(self, image_bgr):
        """Preprocess single image"""
        # Convert to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Resize
        image_resized = cv2.resize(image_rgb, self.target_size, interpolation=cv2.INTER_CUBIC)

        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0

        return image_normalized

    def load_image(self, image_path):
        """Load image from file"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        return image

    def load_images_from_folder(self, folder_path):
        """Load all images from folder"""
        images = []
        paths = []
        folder = Path(folder_path)

        for img_file in folder.glob("*.jpg"):
            try:
                img = self.load_image(img_file)
                images.append(img)
                paths.append(str(img_file))
            except Exception as e:
                print(f"Error loading {img_file}: {e}")

        return images, paths
