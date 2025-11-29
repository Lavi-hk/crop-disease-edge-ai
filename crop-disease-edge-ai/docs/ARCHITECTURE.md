# Model Architecture

## Overview

**Model**: MobileNetV2 with Transfer Learning
**Input**: 224×224×3 RGB images
**Output**: 39 disease classes + 1 healthy class
**Total Parameters**: 3.5M
**Model Size**: 45MB (FP32) → 10MB (INT8 quantized)

## Architecture Details

### Backbone: MobileNetV2

MobileNetV2 is chosen for:
- Efficient mobile deployment (3.5M params vs VGG's 138M)
- Fast inference (160ms on Raspberry Pi)
- High accuracy (95.8% on our dataset)
- Designed for edge devices

### Custom Classification Head

```
Input (224×224×3)
    ↓
MobileNetV2 Base (Pretrained ImageNet)
    ├─ Depthwise Separable Convolutions
    ├─ Inverted Residual Blocks (17×)
    └─ 1280 features → Global Average Pooling
    ↓
Dense(512) → ReLU → Dropout(0.5)
    ↓
Dense(256) → ReLU → Dropout(0.3)
    ↓
Dense(40) → Softmax
    ↓
Output: Disease prediction + confidence
```

## Training Configuration

- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 50
- **Early Stopping**: Patience=5
- **Learning Rate Scheduler**: Exponential decay

## Optimization Techniques

1. **Transfer Learning**: ImageNet pretrained weights
2. **Data Augmentation**: 12+ techniques
3. **INT8 Quantization**: 4x compression
4. **Dropout Regularization**: Prevent overfitting

## Performance

- **Accuracy**: 95.8%
- **Inference Time**: 160ms (RPi 4)
- **Model Size**: 10MB (quantized)
- **Compression Ratio**: 4x
