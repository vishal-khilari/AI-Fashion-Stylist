import torch
import torch.nn as nn
from torchvision import models

# ── Config ────────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Formal', 'Casual', 'Traditional']
NUM_CLASSES = len(CLASS_NAMES)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE    = 224

# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    EfficientNet-B0 pretrained on ImageNet.
    
    Why EfficientNet-B0 over MobileNetV3?
    - Better accuracy/parameter tradeoff
    - Scales better to small datasets via transfer learning
    - Standard classifier head is easier to replace cleanly

    freeze_backbone=True  → only the new classifier head trains  (Stage 1)
    freeze_backbone=False → entire network trains                 (Stage 2)
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # EfficientNet-B0 classifier: [Dropout, Linear(1280 → 1000)]
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # New head is always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Device      : {DEVICE}")
    print(f"Backbone    : EfficientNet-B0")
    print(f"Trainable   : {trainable:,} / {total:,} params")
    return model