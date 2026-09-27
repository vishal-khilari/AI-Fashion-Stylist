# AI Fashion Stylist & Dress Classifier

An AI-powered fashion web app that combines a custom-trained PyTorch image classifier with Google Gemini generative AI to provide outfit predictions, styling feedback, and a structured fashion chatbot.

## Key Features

- **AI Image Classification:** Accurately classifies clothing photos into `Formal`, `Casual`, or `Traditional` styles.
- **Confidence Scoring & Analytics:** Provides per-class probability breakdowns and confidence scores for predictions.
- **Personalized GenAI Stylist:** Integrates Google Gemini (2.5 Flash) to generate creative, energetic fashion compliments and styling tips based on the predicted outfit.
- **Interactive Fashion Chatbot:** Features a dedicated AI chatbot that provides structured, full-outfit recommendations (clothing, footwear, grooming, and accessories).
- **Modern, Responsive Web UI:** A polished frontend with dynamic class-specific metadata (custom emojis, color themes, and descriptions) and intuitive upload interfaces.
- **Serverless-Optimized Inference:** Dual-engine architecture automatically utilizes **ONNX Runtime** for ultra-fast, lightweight inference in serverless environments (like Vercel), while supporting PyTorch fallback for local development.
- **Robust Error Handling:** Seamlessly manages invalid image formats and gracefully handles missing API keys without breaking the application flow.
- **Vercel Deployment Ready:** Pre-configured with an optimized `requirements.txt` and `vercel.json` for immediate deployment to Vercel without exceeding serverless size limits.
- **Comprehensive CLI Tools:** Includes command-line utilities for two-stage model training (EfficientNet-B0 transfer learning), model evaluation, and single-image inference.

## Prerequisites

- Python 3.8 or higher
- `pip`
- Optional: GPU with CUDA support for faster training/inference

## Installation

1. Clone or download the repository and open the project folder.
2. Create and activate a virtual environment:

```bash

(venv) PS C:\Users\Vishal P Khilari\Desktop\github\Major projrct\Tmodel> pip install opencv-python
Collecting opencv-python
  Downloading opencv_python-5.0.0.93-cp37-abi3-win_amd64.whl.metadata (20 kB)
Requirement already satisfied: numpy>=2 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from opencv-python) (2.5.1)
Downloading opencv_python-5.0.0.93-cp37-abi3-win_amd64.whl (44.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.0/44.0 MB 5.1 MB/s eta 0:00:00
Installing collected packages: opencv-python
Successfully installed opencv-python-5.0.0.93

[notice] A new release of pip is available: 24.2 -> 26.2.1
[notice] To update, run: C:\Users\Vishal P Khilari\Desktop\Tmodel\venv\Scripts\python.exe -m pip install --upgrade pip
(venv) PS C:\Users\Vishal P Khilari\Desktop\github\Major projrct\Tmodel> pip install -r requirements.txt
Requirement already satisfied: flask in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from -r requirements.txt (line 1)) (3.1.3)
Requirement already satisfied: pillow in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from -r requirements.txt (line 2)) (12.3.0)
Requirement already satisfied: google-generativeai in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from -r requirements.txt (line 3)) (0.8.6)
Requirement already satisfied: numpy in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from -r requirements.txt (line 4)) (2.5.1)
Requirement already satisfied: onnxruntime in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from -r requirements.txt (line 5)) (1.27.0)
Requirement already satisfied: opencv-python in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from -r requirements.txt (line 6)) (5.0.0.93)
Requirement already satisfied: blinker>=1.9.0 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from flask->-r requirements.txt (line 1)) (1.9.0)
Requirement already satisfied: click>=8.1.3 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from flask->-r requirements.txt (line 1)) (8.4.2)
Requirement already satisfied: itsdangerous>=2.2.0 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from flask->-r requirements.txt (line1)) (2.2.0)
Requirement already satisfied: jinja2>=3.1.2 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from flask->-r requirements.txt (line 1)) (3.1.6)
Requirement already satisfied: markupsafe>=2.1.1 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from flask->-r requirements.txt (line 1)) (3.0.3)
Requirement already satisfied: werkzeug>=3.1.0 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from flask->-r requirements.txt (line 1))(3.1.8)
Requirement already satisfied: google-ai-generativelanguage==0.6.15 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-generativeai->-r requirements.txt (line 3)) (0.6.15)
Requirement already satisfied: google-api-core in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-generativeai->-r requirements.txt (line 3)) (2.32.0)
Requirement already satisfied: google-api-python-client in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-generativeai->-r requirements.txt (line 3)) (2.198.0)
Requirement already satisfied: google-auth>=2.15.0 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-generativeai->-r requirements.txt (line 3)) (2.56.0)
Requirement already satisfied: protobuf in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-generativeai->-r requirements.txt (line 3)) (5.29.6)
Requirement already satisfied: pydantic in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-generativeai->-r requirements.txt (line 3)) (2.13.4)
Requirement already satisfied: tqdm in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-generativeai->-r requirements.txt (line 3)) (4.68.4)
Requirement already satisfied: typing-extensions in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-generativeai->-r requirements.txt (line 3)) (4.16.0)
Requirement already satisfied: proto-plus<2.0.0dev,>=1.22.3 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-ai-generativelanguage==0.6.15->google-generativeai->-r requirements.txt (line 3)) (1.28.1)
Requirement already satisfied: flatbuffers in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from onnxruntime->-r requirements.txt (line 5)) (25.12.19)
Requirement already satisfied: packaging in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from onnxruntime->-r requirements.txt (line 5))(26.2)
Requirement already satisfied: colorama in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from click>=8.1.3->flask->-r requirements.txt (line 1)) (0.4.6)
Requirement already satisfied: googleapis-common-protos<2.0.0,>=1.63.2 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-api-core->google-generativeai->-r requirements.txt (line 3)) (1.75.0)
Requirement already satisfied: requests<3.0.0,>=2.33.0 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-api-core->google-generativeai->-r requirements.txt (line 3)) (2.34.2)
Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-auth>=2.15.0->google-generativeai->-r requirements.txt (line 3)) (0.4.2)
Requirement already satisfied: cryptography>=38.0.3 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-auth>=2.15.0->google-generativeai->-r requirements.txt (line 3)) (49.0.0)
Requirement already satisfied: httplib2<1.0.0,>=0.19.0 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-api-python-client->google-generativeai->-r requirements.txt (line 3)) (0.32.0)
Requirement already satisfied: google-auth-httplib2<1.0.0,>=0.2.0 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-api-python-client->google-generativeai->-r requirements.txt (line 3)) (0.4.0)
Requirement already satisfied: uritemplate<5,>=3.0.1 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-api-python-client->google-generativeai->-r requirements.txt (line 3)) (4.2.0)
Requirement already satisfied: annotated-types>=0.6.0 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from pydantic->google-generativeai->-r requirements.txt (line 3)) (0.7.0)
Requirement already satisfied: pydantic-core==2.46.4 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from pydantic->google-generativeai->-r requirements.txt (line 3)) (2.46.4)
Requirement already satisfied: typing-inspection>=0.4.2 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from pydantic->google-generativeai->-r requirements.txt (line 3)) (0.4.2)
Requirement already satisfied: cffi>=2.0.0 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from cryptography>=38.0.3->google-auth>=2.15.0->google-generativeai->-r requirements.txt (line 3)) (2.1.0)
Requirement already satisfied: grpcio<2.0.0,>=1.41.0 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-ai-generativelanguage==0.6.15->google-generativeai->-r requirements.txt (line 3)) (1.82.1)
Requirement already satisfied: grpcio-status<2.0.0,>=1.41.0 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-ai-generativelanguage==0.6.15->google-generativeai->-r requirements.txt (line 3)) (1.71.2)
Requirement already satisfied: pyparsing<4,>=3.1 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from httplib2<1.0.0,>=0.19.0->google-api-python-client->google-generativeai->-r requirements.txt (line 3)) (3.3.2)
Requirement already satisfied: pyasn1<0.7.0,>=0.6.1 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth>=2.15.0->google-generativeai->-r requirements.txt (line 3)) (0.6.4)
Requirement already satisfied: charset_normalizer<4,>=2 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from requests<3.0.0,>=2.33.0->google-api-core->google-generativeai->-r requirements.txt (line 3)) (3.4.9)
Requirement already satisfied: idna<4,>=2.5 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from requests<3.0.0,>=2.33.0->google-api-core->google-generativeai->-r requirements.txt (line 3)) (3.18)
Requirement already satisfied: urllib3<3,>=1.26 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from requests<3.0.0,>=2.33.0->google-api-core->google-generativeai->-r requirements.txt (line 3)) (2.7.0)
Requirement already satisfied: certifi>=2023.5.7 in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from requests<3.0.0,>=2.33.0->google-api-core->google-generativeai->-r requirements.txt (line 3)) (2026.6.17)
Requirement already satisfied: pycparser in c:\users\vishal p khilari\desktop\tmodel\venv\lib\site-packages (from cffi>=2.0.0->cryptography>=38.0.3->google-auth>=2.15.0->google-generativeai->-r requirements.txt (line 3)) (3.0)

[notice] A new release of pip is available: 24.2 -> 26.2.1
[notice] To update, run: C:\Users\Vishal P Khilari\Desktop\Tmodel\venv\Scripts\python.exe -m pip install --upgrade pip
(venv) PS C:\Users\Vishal P Khilari\Desktop\github\Major projrct\Tmodel> python app.py
Traceback (most recent call last):
  File "C:\Users\Vishal P Khilari\Desktop\github\Major projrct\Tmodel\app.py", line 21, in <module>
    from camera_manager import CameraManager, scan_available_cameras
  File "C:\Users\Vishal P Khilari\Desktop\github\Major projrct\Tmodel\camera_manager.py", line 18, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'
(venv) PS C:\Users\Vishal P Khilari\Desktop\github\Major projrct\Tmodel> 
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install the required packages:

- For local development and model training (includes PyTorch & torchvision):
  ```bash
  pip install -r requirements-dev.txt
  ```

- For lightweight web app running only (using ONNX Runtime):
  ```bash
  pip install -r requirements.txt
  ```

4. (Optional) Install additional packages for dataset image downloads:

```bash
pip install icrawler
```

5. Add your Gemini API key:

- Create `api_key.txt` in the project root, or
- Set the environment variable `GEMINI_API_KEY`

Put only the raw API key text in the file.

6. Confirm that `dress_classifier.pth` exists in the project root.

## Running the App

Start the Flask web server:

```bash
python app.py
```

Open the browser at:

```text
http://localhost:5000
```

## Web Pages

- `/` : Home page
- `/upload/<category>` : Image upload page for a specific outfit category
- `/chatbot` : AI stylist chat interface
- `/health` : Simple health check endpoint

## Gemini Integration

The app uses Google Gemini if a valid API key is provided via:

- `api_key.txt`, or
- `GEMINI_API_KEY`

If the key is configured, the app returns creative styling feedback for uploads and structured JSON outfit suggestions in the chatbot.

## CLI Usage

The `main.py` script supports multiple modes:

```bash
python main.py train
python main.py eval
python main.py predict <image_path>
```

- `train` : Run the two-stage training pipeline and save the best model to `dress_classifier.pth`
- `eval` : Evaluate the validation set and print a confusion matrix
- `predict <image_path>` : Run a single image through the saved model from the command line

### Real-Time Live USB Mobile Camera

Run the real-time desktop classifier using your USB-connected mobile phone camera:

```bash
# Auto-detect camera:
python live_camera.py

# Or specify a mobile camera index or stream:
python live_camera.py --source 1
python live_camera.py --source http://127.0.0.1:8080/video
python live_camera.py --source mock   # synthetic test source
```

See [USB_MOBILE_CAMERA_SETUP.md](USB_MOBILE_CAMERA_SETUP.md) for full details on connecting via DroidCam, Iriun, IP Webcam + ADB, or Android 14+ Webcam.

You can also run single-image inference directly with:

```bash
python SingleImageInference.py <image_path>
```

## Dataset Structure

The project expects training and validation data under `dataset/` with this layout:

```text
dataset/
  train/
    Formal/
    Casual/
    Traditional/
  val/
    Formal/
    Casual/
    Traditional/
```

## Core Modules

- `app.py` : Flask server, image upload/predict endpoint, USB live camera stream & classification, GenAI feedback, chatbot API
- `camera_manager.py` : Thread-safe USB / Mobile camera capture engine with bounded reconnection and mock testing
- `live_camera.py` : Standalone real-time desktop HUD classifier for USB mobile camera feed
- `main.py` : CLI entry point for training, evaluation, and prediction
- `initialisation.py` : Model builder, device config, class labels
- `Transforms.py` : Training/validation/inference image transformations
- `SaveAndReload.py` : Save/load model checkpoint utilities
- `SingleImageInference.py` : Single-image prediction and validation utilities
- `two_stage_training.py` : Training pipeline with stage 1 head training and stage 2 fine-tuning
- `dataset.py` : Custom `DressDataset` loader for train/val image folders
- `download_images.py` : Optional script to download fashion images via Bing Image Crawler

## Notes

- The app is built around a pre-trained `EfficientNet-B0` backbone.
- The classifier currently supports three categories: `Formal`, `Casual`, and `Traditional`.
- The `download_images.py` script includes a larger query set, but the model loader only trains on the three supported classes.

## Troubleshooting

- If the model file is missing, `app.py` will log an error and fail to start prediction.
- If Gemini is not configured, the app still works for image classification but returns a message prompting for the API key.
- For dataset issues, verify folder names and image extensions (`.jpg`, `.jpeg`, `.png`, `.webp`).

## GitHub and Vercel Deployment

### GitHub

- Initialize the repository locally with `git init`.
- Add all project files and commit.
- Set a remote such as `origin` and push to GitHub.
- `api_key.txt`, `dataset/`, and PyTorch checkpoint files (`*.pth`, `*.pt`) are excluded by `.gitignore` to keep the repo clean and avoid leaking secrets.
- **Note**: The self-contained ONNX model `dress_classifier.onnx` is tracked and pushed to GitHub so Vercel can access it directly.

### Vercel

- The Flask app in `app.py` has a dual-engine system. It will automatically load the model using `onnxruntime` if available (recommended for Vercel/CPU serverless functions), and fall back to PyTorch only if ONNX is not present.
- The `requirements.txt` has been optimized specifically for Vercel Serverless Functions by removing `torch` and `torchvision` (which exceed size limits) and using `onnxruntime` + `numpy` instead.
- Set the `GEMINI_API_KEY` environment variable in your Vercel project dashboard to enable AI stylist chatbot features.

### Quick Start

```bash
git init
git add .
git commit -m "Initialize repo with ONNX support for Vercel deployment"
```

Then add your GitHub remote and push:

```bash
git remote add origin https://github.com/<your-user>/<your-repo>.git
git branch -M main
git push -u origin main
```
