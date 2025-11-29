# Results & Performance

## Overall Metrics

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 95.8% |
| Top-5 Accuracy | 99.2% |
| Precision (Macro) | 94.2% |
| Recall (Macro) | 94.8% |
| F1-Score (Macro) | 0.945 |

## Inference Performance

| Device | Model | Latency | FPS |
|--------|-------|---------|-----|
| Raspberry Pi 4 | INT8 | 160ms | 6 |
| ESP32-CAM | INT8 | 350ms | 3 |
| Android Flagship | INT8 | 60ms | 17 |
| Jetson Nano | INT8 | 100ms | 10 |
| Cloud (CPU) | FP32 | 40ms | 25 |

## Model Size Comparison

| Format | Size | Compression | Accuracy |
|--------|------|-------------|----------|
| SavedModel (FP32) | 45MB | 1x | 95.8% |
| TFLite (FP32) | 44MB | 1x | 95.8% |
| TFLite (FP16) | 22MB | 2x | 95.7% |
| TFLite (INT8) | 10MB | 4x | 95.5% |

## Real-World Testing

- **Farms Tested**: 15
- **Images Tested**: 200+
- **Field Accuracy**: 94.2%
- **Most Confident**: Healthy (99.2%), Apple scab (97.2%)

## Dataset Distribution

**Total Images**: 54,000
- Training: 37,800 (70%)
- Validation: 8,100 (15%)
- Test: 8,100 (15%)

**Crops**: 14
- Apple, Blueberry, Cherry, Corn, Grape, Orange
- Peach, Pepper, Potato, Raspberry, Soybean
- Squash, Strawberry, Tomato, Wheat

**Diseases**: 39 classes
- Bacterial (3), Fungal (8), Viral (4)
- Nutrient deficiency (8), Environmental (2)
- Healthy (1 class per crop)
