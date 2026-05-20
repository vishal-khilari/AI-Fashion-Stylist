"""
download_images.py
------------------
Downloads training and validation images using Bing Image Crawler.

Targets:
  Train: 6 queries × 50 images = ~300 per class
  Val:   2 queries × 30 images = ~60  per class

Tips if downloads are cut short:
  - Add a VPN or rotate IPs (Bing rate-limits aggressively)
  - Or download fashion datasets from Kaggle directly:
      https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
"""

from icrawler.builtin import BingImageCrawler
import os
import time

dataset = {
    'train': {
        'Formal': [
            'men formal suit office professional',
            'women formal dress office business',
            'business formal blazer trousers',
            'formal shirt tie jacket professional',
            'men formal wear suit tie',
            'women formal pencil skirt blazer',
        ],
        'Casual': [
            'men casual t-shirt jeans street',
            'women casual everyday outfit',
            'casual hoodie sweatpants relaxed',
            'casual polo shirt chinos men',
            'women casual denim jacket outfit',
            'relaxed casual streetwear outfit',
        ],
        'Traditional': [
            'men kurta pajama indian traditional',
            'women saree indian traditional outfit',
            'salwar kameez ethnic wear indian',
            'sherwani traditional indian men',
            'lehenga choli traditional indian women',
            'dhoti angavastram traditional south indian',
        ],
        'Sportswear': [
            'men gym workout outfit athletic',
            'women sports leggings top workout',
            'football jersey tracksuit sportswear',
            'running athletic wear men',
            'basketball sports uniform jersey',
            'cycling sportswear compression outfit',
        ],
    },
    'val': {
        'Formal':      ['formal business attire professional', 'formal dress suit office wear'],
        'Casual':      ['casual clothing everyday style jeans', 'casual shirt outfit relaxed'],
        'Traditional': ['traditional indian saree outfit', 'kurta ethnic wear india men'],
        'Sportswear':  ['sportswear athlete gym outfit', 'sports jersey tracksuit running'],
    }
}

TRAIN_PER_QUERY = 50   # target ~300/class train
VAL_PER_QUERY   = 30   # target ~60/class  val
DELAY_BETWEEN   = 3    # seconds between queries (reduces rate-limit hits)

def download_all():
    for split, classes in dataset.items():
        per_q = TRAIN_PER_QUERY if split == 'train' else VAL_PER_QUERY
        for class_name, queries in classes.items():
            save_dir = os.path.join('dataset', split, class_name)
            os.makedirs(save_dir, exist_ok=True)

            for query in queries:
                print(f"  [{split}/{class_name}] '{query}' → {per_q} images")
                try:
                    crawler = BingImageCrawler(
                        storage={'root_dir': save_dir},
                        feeder_threads=1,
                        parser_threads=1,
                        downloader_threads=4,
                    )
                    crawler.crawl(keyword=query, max_num=per_q)
                except Exception as e:
                    print(f"  [ERROR] {e}")
                time.sleep(DELAY_BETWEEN)

    # ── Final count ───────────────────────────────────────────────────────────
    print("\n── Final image counts ──")
    for split in ['train', 'val']:
        for cls in ['Formal', 'Casual', 'Traditional', 'Sportswear']:
            path  = os.path.join('dataset', split, cls)
            count = len([f for f in os.listdir(path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) \
                    if os.path.isdir(path) else 0
            status = "✓" if count >= (200 if split == 'train' else 40) else "⚠ LOW"
            print(f"  {split}/{cls:<14}: {count:>4} images  {status}")

    print("\nDone! If counts are low, see the Kaggle link in the docstring.")


if __name__ == '__main__':
    download_all()