"""
SingleImageInference.py
-----------------------
Two modes:
  1. predict_style(image_path, model)  — single image → class + confidence
  2. evaluate_folder(folder, model)    — runs on a whole val split and prints a confusion matrix
"""

import torch
import torch.nn as nn
from PIL import Image
from Transforms import inference_transform
from initialisation import CLASS_NAMES, DEVICE
from SaveAndReload import load_model


# ── Single image prediction ───────────────────────────────────────────────────
def predict_style(image_path: str, model: nn.Module) -> dict:
    """
    Returns:
        {
          'predicted_class': 'Formal',
          'confidence': 0.923,
          'all_scores': {'Formal': 0.923, 'Casual': 0.05, ...}
        }
    """
    model.eval()
    img    = Image.open(image_path).convert('RGB')
    tensor = inference_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_idx = probs.argmax().item()
    return {
        'predicted_class': CLASS_NAMES[top_idx],
        'confidence':      round(probs[top_idx].item(), 4),
        'all_scores':      {c: round(probs[i].item(), 4) for i, c in enumerate(CLASS_NAMES)}
    }


# ── Per-class confusion matrix ────────────────────────────────────────────────
def evaluate_folder(root_dir: str, split: str, model: nn.Module):
    """
    Walks dataset/split/ClassName folders and prints:
      - Overall accuracy
      - Per-class accuracy
      - Confusion matrix
    """
    import os
    from collections import defaultdict

    model.eval()
    n   = len(CLASS_NAMES)
    cm  = [[0] * n for _ in range(n)]   # cm[true][pred]

    for true_idx, class_name in enumerate(CLASS_NAMES):
        folder = os.path.join(root_dir, split, class_name)
        if not os.path.isdir(folder):
            print(f"[WARNING] {folder} not found, skipping.")
            continue

        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            result   = predict_style(os.path.join(folder, fname), model)
            pred_idx = CLASS_NAMES.index(result['predicted_class'])
            cm[true_idx][pred_idx] += 1

    # ── Print confusion matrix ────────────────────────────────────────────────
    col_w = 14
    print("\n── Confusion Matrix (rows=True, cols=Predicted) ──")
    header = f"{'':>{col_w}}" + "".join(f"{c:>{col_w}}" for c in CLASS_NAMES)
    print(header)
    print("-" * len(header))

    total_correct, total = 0, 0
    for i, row_name in enumerate(CLASS_NAMES):
        row_total   = sum(cm[i])
        row_correct = cm[i][i]
        total_correct += row_correct
        total         += row_total
        pct = f"({row_correct/row_total:.0%})" if row_total else "(n/a)"
        row_str = f"{row_name:>{col_w}}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(n))
        print(f"{row_str}   {pct}")

    print("-" * len(header))
    print(f"\nOverall accuracy: {total_correct}/{total} = {total_correct/total:.1%}")

    # ── Per-class accuracy ────────────────────────────────────────────────────
    print("\n── Per-class accuracy ──")
    for i, c in enumerate(CLASS_NAMES):
        row_total = sum(cm[i])
        acc = cm[i][i] / row_total if row_total else 0
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {c:<14} {bar}  {acc:.1%}")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    model = load_model('dress_classifier.pth')

    if len(sys.argv) > 1:
        # Usage: python SingleImageInference.py path/to/image.jpg
        image_path = sys.argv[1]
        result     = predict_style(image_path, model)
        print(f"\nImage      : {image_path}")
        print(f"Prediction : {result['predicted_class']}  ({result['confidence']:.1%} confidence)")
        print("All scores :")
        for cls, score in result['all_scores'].items():
            bar = "█" * int(score * 30)
            print(f"  {cls:<14} {bar:<30}  {score:.1%}")
    else:
        # No argument: run full evaluation on val set
        evaluate_folder('dataset', 'val', model)