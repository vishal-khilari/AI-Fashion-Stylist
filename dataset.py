from torch.utils.data import Dataset
from PIL import Image
import os

CLASS_NAMES = ['Formal', 'Casual', 'Traditional']

class DressDataset(Dataset):
    """
    Folder structure expected:
        root_dir/
            train/
                Formal/      *.jpg, *.jpeg, *.png
                Casual/
                Traditional/
                Sportswear/
            val/
                ...
    """
    def __init__(self, root_dir: str, split: str, transform=None):
        self.transform = transform
        self.samples   = []

        split_dir = os.path.join(root_dir, split)
        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"[WARNING] Missing folder: {class_dir}")
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.samples.append(
                        (os.path.join(class_dir, fname), label_idx)
                    )

        print(f"[{split}] Loaded {len(self.samples)} images across {len(CLASS_NAMES)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def class_counts(self):
        """Returns a dict of {class_name: count} for diagnostics."""
        counts = {c: 0 for c in CLASS_NAMES}
        for _, label in self.samples:
            counts[CLASS_NAMES[label]] += 1
        return counts