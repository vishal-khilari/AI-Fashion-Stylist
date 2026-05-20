# AI Fashion Stylist & Dress Classifier

An AI-powered fashion web app that combines a custom-trained PyTorch image classifier with Google Gemini generative AI to provide outfit predictions, styling feedback, and a structured fashion chatbot.

## Key Features

- **Image classification** of clothing photos into `Formal`, `Casual`, or `Traditional`.
- **Confidence scoring** with per-class probability breakdown.
- **Personalized GenAI feedback** when `Gemini` API access is configured.
- **Fashion stylist chatbot** that returns structured outfit recommendations in JSON.
- **Web UI** with upload and chatbot pages.
- **CLI utilities** for training, evaluation, and single-image prediction.
- **Two-stage training pipeline** using EfficientNet-B0 with transfer learning.

## Prerequisites

- Python 3.8 or higher
- `pip`
- Optional: GPU with CUDA support for faster training/inference

## Installation

1. Clone or download the repository and open the project folder.
2. Create and activate a virtual environment:

```bash
python -m venv venv
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

- `app.py` : Flask server, image upload/predict endpoint, GenAI feedback, chatbot API
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
