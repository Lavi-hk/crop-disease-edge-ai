"""
Lightweight TFLite Inference Engine
"""
import numpy as np
import tensorflow as tf
import cv2

class InferenceEngine:
    """Inference engine for TFLite models"""

    def __init__(self, model_path, class_labels=None):
        """Initialize with TFLite model"""
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.class_labels = class_labels or []

    def _preprocess(self, image, target_size=None):
        """Preprocess image for inference"""
        if target_size is None:
            target_size = self.input_details[0]["shape"][1:3]
        img = cv2.resize(image, tuple(target_size))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, image_bgr):
        """Run inference on image"""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inp = self._preprocess(image_rgb)

        self.interpreter.set_tensor(self.input_details[0]["index"], inp)
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        label = self.class_labels[idx] if self.class_labels else f"class_{idx}"

        return {
            "class_index": idx, 
            "class_name": label, 
            "confidence": conf, 
            "all_predictions": preds.tolist()
        }
