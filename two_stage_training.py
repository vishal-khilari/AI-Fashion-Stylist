import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import os

from initialisation import build_model, DEVICE, NUM_CLASSES, CLASS_NAMES
from Transforms import train_transform, val_transform
from dataset import DressDataset

# ── Config ────────────────────────────────────────────────────────────────────
PATIENCE      = 7
BATCH_SIZE    = 16
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 25

# ── Data ──────────────────────────────────────────────────────────────────────
train_dataset = DressDataset('dataset', split='train', transform=train_transform)
val_dataset   = DressDataset('dataset', split='val',   transform=val_transform)

# Weighted sampler: gives rarer classes more chance to appear each epoch
# Fixes the slight class imbalance (85 vs 93 images/class)
def make_weighted_sampler(dataset):
    counts  = dataset.class_counts()
    weights = [1.0 / counts[CLASS_NAMES[label]] for _, label in dataset.samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=make_weighted_sampler(train_dataset),
    num_workers=0
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Loss: label smoothing reduces overconfident predictions ───────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ── One epoch helper ──────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer=None, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total

# ── Training entry point ──────────────────────────────────────────────────────
def train(save_path='dress_classifier.pth'):
    model = build_model(num_classes=NUM_CLASSES, freeze_backbone=True)

    # ── Stage 1: train only the new classifier head ───────────────────────────
    print("\n── Stage 1: training head only ──")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    for epoch in range(STAGE1_EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,              training=False)
        print(f"  Epoch {epoch+1:02d}/{STAGE1_EPOCHS} | "
              f"train loss {tr_loss:.4f}  acc {tr_acc:.3f} | "
              f"val loss {vl_loss:.4f}  acc {vl_acc:.3f}")

    # ── Stage 2: fine-tune the full network at a lower LR ─────────────────────
    print("\n── Stage 2: fine-tuning full network ──")
    for param in model.parameters():
        param.requires_grad = True

    # Lower LR (5e-5 vs 1e-4) — backbone weights are pretrained, be gentle
    optimizer         = torch.optim.Adam(model.parameters(), lr=5e-5)
    scheduler         = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE2_EPOCHS)
    best_val_acc      = 0.0
    epochs_no_improve = 0

    for epoch in range(STAGE2_EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,              training=False)
        scheduler.step()

        if vl_acc > best_val_acc:
            best_val_acc      = vl_acc
            epochs_no_improve = 0
            torch.save({
                'epoch':            epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc':          vl_acc,
                'class_names':      CLASS_NAMES,
            }, save_path)
            print(f"  Epoch {epoch+1:02d}/{STAGE2_EPOCHS} | "
                  f"train acc {tr_acc:.3f} | val acc {vl_acc:.3f}  ← saved ✓")
        else:
            epochs_no_improve += 1
            print(f"  Epoch {epoch+1:02d}/{STAGE2_EPOCHS} | "
                  f"train acc {tr_acc:.3f} | val acc {vl_acc:.3f}  "
                  f"(no improvement {epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete.  Best val acc : {best_val_acc:.3f}")
    print(f"Model saved to     : {save_path}")
    return model


if __name__ == '__main__':
    print("── Dataset diagnostics ──")
    for split in ['train', 'val']:
        for cls in CLASS_NAMES:
            path  = os.path.join('dataset', split, cls)
            count = len([f for f in os.listdir(path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]) \
                    if os.path.isdir(path) else 0
            status = f"{count} images" if count else "FOLDER NOT FOUND"
            print(f"  {split}/{cls}: {status}")
    train()