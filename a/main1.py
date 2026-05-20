import os
import sys
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ✅ Convert to highest quality image safely
def get_high_quality_url(url):
    try:
        # Only modify URLs that look like standard pin image formats
        if "pinimg.com" in url and ("236x" in url or "474x" in url or "736x" in url):
            parts = url.split("/")
            return "https://i.pinimg.com/originals/" + "/".join(parts[-4:])
    except Exception:
        pass
    return url

def download_pinterest(url, folder="dataset/treditionalwomen", max_images=100):
    os.makedirs(folder, exist_ok=True)

    options = webdriver.ChromeOptions()

    # ============================
    # STABILITY FLAGS (Clean Profile)
    # ============================
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--start-maximized")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    driver.get(url)
    
    # Give yourself time to log into Pinterest manually so it doesn't block the scrolling
    print("⏳ Waiting 30 seconds... If Pinterest asks you to log in, do it now!")
    time.sleep(30) 

    print("Scrolling to collect image links...")

    image_urls = set()

    # scroll more = more images
    for _ in range(40):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        images = driver.find_elements(By.TAG_NAME, "img")

        for img in images:
            try:
                src = img.get_attribute("src")

                # Filter out profile pictures/icons; only grab actual pins
                if src and "i.pinimg.com" in src and ("236x" in src or "736x" in src):
                    high_res = get_high_quality_url(src)
                    image_urls.add(high_res)

            except Exception:
                pass

        print(f"Collected: {len(image_urls)}")

        if len(image_urls) >= max_images:
            break

    driver.quit()

    print(f"\nAttempting to download {min(len(image_urls), max_images)} HIGH QUALITY images...")

    # 👇 User-Agent header to prevent Pinterest from blocking the download
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    count = 0
    # Convert to list and slice to ensure we don't exceed max_images
    for img_url in list(image_urls)[:max_images]:
        try:
            response = requests.get(img_url, headers=headers, timeout=10)

            if response.status_code != 200:
                print(f"❌ Failed (Status {response.status_code}): {img_url}")
                continue

            with open(f"{folder}/{count}.jpg", "wb") as f:
                f.write(response.content)

            count += 1
            print(f"✅ Downloaded {count}")

        except Exception as e:
            print(f"⚠️ Error with {img_url}: {e}")

    print(f"\n🎉 Done. Successfully downloaded {count} images.")

# ============================
# RUN
# ============================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <pinterest_url> [max_images]")
        sys.exit()

    url = sys.argv[1]
    max_images = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    download_pinterest(url, "dataset/treditionalwomen", max_images)