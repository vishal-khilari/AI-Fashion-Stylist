import torch
import torch.nn as nn
from initialisation import build_model, DEVICE, CLASS_NAMES

def save_model(model: nn.Module, path: str = 'dress_classifier.pth', epoch: int = 0, val_acc: float = 0.0):
    """Save model weights + metadata to disk."""
    torch.save({
        'epoch':            epoch,
        'model_state_dict': model.state_dict(),
        'val_acc':          val_acc,
        'class_names':      CLASS_NAMES,
    }, path)
    print(f"Model saved → {path}  (epoch {epoch}, val_acc {val_acc:.3f})")


def load_model(checkpoint_path: str) -> nn.Module:
    """Load a saved checkpoint and return a ready-to-use model."""
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    model = build_model(num_classes=len(ckpt['class_names']), freeze_backbone=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Trained for {ckpt.get('epoch', '?')} epochs | best val acc: {ckpt.get('val_acc', '?'):.3f}")
    return model