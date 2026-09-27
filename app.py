"""
app.py — Enhanced Flask Web Server for AI Fashion Stylist & Dress Classifier
-----------------------------------------------------------------------------
Features:
- Dual AI inference engine: ONNX Runtime (fast, serverless) + PyTorch fallback
- Real-time USB / Mobile Camera live streaming & automatic frame classification
- Localhost security guard for camera management endpoints
- Dynamic styling feedback powered by Google Gemini GenAI
- Interactive Fashion Chatbot API with structured JSON outfit recommendations
"""

import io
import os
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from PIL import Image, UnidentifiedImageError
from urllib.parse import urlparse

# Camera manager for USB / Mobile / WebRTC capture
from camera_manager import CameraManager, scan_available_cameras

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

# Try importing PyTorch for local fallback
try:
    import torch
    from initialisation import CLASS_NAMES, DEVICE
    from Transforms import inference_transform
    from SaveAndReload import load_model
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    CLASS_NAMES = ['Formal', 'Casual', 'Traditional']
    DEVICE = 'cpu'
    inference_transform = None
    load_model = None

app = Flask(__name__)

# Model configuration
MODEL_PTH_PATH = 'dress_classifier.pth'
MODEL_ONNX_PATH = 'dress_classifier.onnx'

model = None
ort_session = None

# Initialize real-time camera manager
camera_manager = CameraManager()


def get_model():
    """Lazy load the model (ONNX or PyTorch) to ensure it's ready when needed."""
    global model, ort_session

    # 1. Try ONNX model first (recommended for Vercel/CPU serverless environments)
    if HAS_ORT and os.path.exists(MODEL_ONNX_PATH):
        if ort_session is None:
            logger.info("Loading dress classifier model using ONNX Runtime...")
            ort_session = ort.InferenceSession(MODEL_ONNX_PATH)
            logger.info("ONNX model loaded successfully!")
        return ort_session

    # 2. Fall back to PyTorch if available
    if HAS_TORCH:
        if model is None:
            if not os.path.exists(MODEL_PTH_PATH):
                logger.error(f"Model file not found at {MODEL_PTH_PATH}")
                raise FileNotFoundError(f"Model file {MODEL_PTH_PATH} not found.")

            logger.info("Loading dress classifier model using PyTorch...")
            model = load_model(MODEL_PTH_PATH)
            model.to(DEVICE)
            model.eval()
            logger.info(f"Model loaded successfully on {DEVICE}!")
        return model

    raise RuntimeError("Neither ONNX Runtime + ONNX model nor PyTorch + PyTorch model could be loaded.")


def preprocess_image_numpy(img: Image.Image) -> np.ndarray:
    """Helper to preprocess images for ONNX Runtime using NumPy & PIL (no torchvision dependency)."""
    img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)
    img_data = np.array(img_resized).astype(np.float32) / 255.0
    if img_data.ndim == 2:
        img_data = np.stack([img_data] * 3, axis=-1)
    elif img_data.shape[2] == 4:
        img_data = img_data[:, :, :3]
    img_data = img_data.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_data = (img_data - mean) / std
    return np.expand_dims(img_data, axis=0).astype(np.float32)


# Class-specific metadata for UI enhancement
CLASS_META = {
    'Formal': {
        'emoji': '👔',
        'desc': 'Sophisticated attire: Suits, blazers, and elegant dresses.',
        'color': '#4a90d9',
    },
    'Casual': {
        'emoji': '👕',
        'desc': 'Comfortable everyday wear: T-shirts, jeans, and hoodies.',
        'color': '#5cb85c',
    },
    'Traditional': {
        'emoji': '🥻',
        'desc': 'Cultural heritage: Ethnic wear like Sarees, Kurtas, and Sherwanis.',
        'color': '#c07bd0',
    },
}

# Fallback metadata for any classes not explicitly defined above
DEFAULT_META = {
    'emoji': '👗',
    'desc': 'Clothing item detected.',
    'color': '#c9a96e',
}


def is_local_request() -> bool:
    """Check if the incoming HTTP request is originating from localhost."""
    remote = request.remote_addr
    return remote in ('127.0.0.1', 'localhost', '::1', 'testclient')


def validate_camera_source(source) -> bool:
    """
    Validate that the requested camera source is safe:
    - Integer index (e.g., 0, 1, 2)
    - 'mock' test source
    - Local loopback stream URL (127.0.0.1, localhost) or standard rtsp/http camera stream
    """
    if isinstance(source, int):
        return 0 <= source <= 10
    if isinstance(source, str):
        if source.isdigit():
            return 0 <= int(source) <= 10
        if source.lower() == 'mock':
            return True
        try:
            parsed = urlparse(source)
            if parsed.scheme in ('http', 'https', 'rtsp'):
                # Safe camera stream URL
                return True
        except Exception:
            return False
    return False


def run_inference_on_pil_image(img: Image.Image) -> dict:
    """
    Core AI prediction pipeline for a PIL Image.
    Used uniformly for file uploads, live USB camera frames, and API calls.
    """
    # 1. Ensure RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 2. Get Model
    current_model = get_model()

    # 3. Inference
    if ort_session is not None:
        # ONNX Runtime inference
        numpy_tensor = preprocess_image_numpy(img)
        input_name = ort_session.get_inputs()[0].name
        logits = ort_session.run(None, {input_name: numpy_tensor})[0]
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs_arr = (e_x / e_x.sum(axis=1, keepdims=True))[0]
        top_idx = int(np.argmax(probs_arr))
        top_prob = float(probs_arr[top_idx])
    else:
        # PyTorch inference
        tensor = inference_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = current_model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        probs_arr = probs.cpu().numpy()
        top_idx = probs.argmax().item()
        top_prob = probs[top_idx].item()

    # 4. Process results
    top_class = CLASS_NAMES[top_idx]
    all_scores = []
    for i, class_name in enumerate(CLASS_NAMES):
        meta = CLASS_META.get(class_name, DEFAULT_META)
        all_scores.append({
            'class': class_name,
            'score': float(probs_arr[i] * 100),
            'emoji': meta['emoji'],
            'desc': meta['desc'],
            'color': meta['color']
        })

    all_scores.sort(key=lambda x: x['score'], reverse=True)
    top_meta = CLASS_META.get(top_class, DEFAULT_META)

    # 5. Gemini Styling Advice Integration
    ai_feedback = None
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key and os.path.exists("api_key.txt"):
        try:
            with open("api_key.txt", "r") as f:
                gemini_api_key = f.read().strip()
        except Exception:
            pass

    if gemini_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            gen_model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = (
                f"Act as an expert fashion stylist. The user has an outfit that our AI classified "
                f"as {top_class} with {round(top_prob * 100, 1)}% confidence. "
                "Give a short, energetic 2-sentence fashion compliment or styling tip. Be creative and hype them up!"
            )
            gen_response = gen_model.generate_content(prompt)
            ai_feedback = gen_response.text
        except Exception as e:
            logger.warning(f"Failed to generate AI feedback: {e}")
            ai_feedback = "You look fantastic! (GenAI stylist is currently unavailable due to an error)."
    else:
        ai_feedback = "Set your GEMINI_API_KEY environment variable to get personalized GenAI styling advice!"

    return {
        'success': True,
        'predicted_class': top_class,
        'confidence': round(top_prob * 100, 1),
        'emoji': top_meta['emoji'],
        'desc': top_meta['desc'],
        'color': top_meta['color'],
        'scores': all_scores,
        'ai_feedback': ai_feedback
    }


# ==============================================================================
# Web Routes
# ==============================================================================

@app.route('/')
def index():
    """Serve the main application home page."""
    return render_template('home.html')


@app.route('/upload/<category>')
def upload(category):
    """Serve the image upload page for a specific category."""
    return render_template('upload.html', category=category)


@app.route('/chatbot')
def chatbot():
    """Serve the chatbot interface."""
    return render_template('chatbot.html')


@app.route('/health')
def health():
    """System health and active camera status endpoint."""
    cam_status = camera_manager.get_status()
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'camera': cam_status
    })


# ==============================================================================
# Camera Management & Live Stream APIs
# ==============================================================================

@app.route('/api/camera/devices', methods=['GET'])
def get_camera_devices():
    """Scan and return all available camera devices (indices, local streams, mock)."""
    devices = scan_available_cameras()
    return jsonify({
        'success': True,
        'devices': devices,
        'current_source': str(camera_manager.source),
        'status': camera_manager.status
    })


@app.route('/api/camera/status', methods=['GET'])
def get_camera_status():
    """Get the current live camera status, resolution, and FPS."""
    return jsonify(camera_manager.get_status())


@app.route('/api/camera/select', methods=['POST'])
def select_camera_source():
    """
    Select or switch the camera source (e.g. 0, 1, 'http://127.0.0.1:8080/video', 'mock').
    Restricted to localhost / safe sources to prevent SSRF.
    """
    if not is_local_request():
        return jsonify({'error': 'Unauthorized: Camera control is restricted to localhost.'}), 403

    data = request.get_json(silent=True) or {}
    source = data.get('source')

    if source is None:
        return jsonify({'error': 'No camera source specified'}), 400

    if not validate_camera_source(source):
        return jsonify({'error': 'Invalid or disallowed camera source'}), 400

    # If it's a numeric string, convert to int
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    success = camera_manager.switch_source(source)
    status_info = camera_manager.get_status()

    return jsonify({
        'success': success,
        'source': str(source),
        'status': status_info
    })


@app.route('/video_feed')
def video_feed():
    """
    MJPEG live video stream route.
    Streams real-time frames from the USB mobile camera / active source.
    """
    # Ensure camera manager is running
    if not camera_manager.is_running:
        camera_manager.start()

    return Response(
        camera_manager.generate_mjpeg_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/camera/capture_and_predict', methods=['POST'])
def capture_and_predict():
    """
    Grab the freshest frame directly from the active USB / Mobile camera,
    pass it through the AI classification model, and return the prediction.
    """
    # Check if camera manager is running
    if not camera_manager.is_running:
        camera_manager.start()

    frame_pil = camera_manager.get_current_frame(as_pil=True)
    if frame_pil is None:
        status_info = camera_manager.get_status()
        err_msg = status_info.get('error_message') or f"No live frame available. Camera status: {status_info.get('status')}."
        return jsonify({'error': err_msg, 'status': status_info}), 400

    try:
        result = run_inference_on_pil_image(frame_pil)
        return jsonify(result)
    except Exception as e:
        logger.exception("Error during camera live prediction")
        return jsonify({'error': f"Inference failed: {str(e)}"}), 500


# ==============================================================================
# Standard Image Upload & Chat APIs (Preserved & Backward Compatible)
# ==============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload (multipart/form-data) and return style predictions."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img_bytes = file.read()
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except UnidentifiedImageError:
            return jsonify({'error': 'Invalid image format. Please upload a JPG, PNG, or WEBP.'}), 400

        result = run_inference_on_pil_image(img)
        return jsonify(result)

    except Exception as e:
        logger.exception("Error during file prediction")
        return jsonify({'error': f"An internal error occurred: {str(e)}"}), 500


@app.route('/chat_api', methods=['POST'])
def chat_api():
    """Handle chatbot messages using Gemini."""
    data = request.get_json(silent=True) or {}
    user_message = data.get('message')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key and os.path.exists("api_key.txt"):
        try:
            with open("api_key.txt", "r") as f:
                gemini_api_key = f.read().strip()
        except Exception:
            pass

    if not gemini_api_key:
        return jsonify({
            'reply': "Please set your GEMINI_API_KEY environment variable or create an api_key.txt file to chat with the stylist!"
        })

    try:
        import json
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        gen_model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""Act as an expert AI fashion stylist. The user is asking: '{user_message}'.
Give them exactly 3 complete styling suggestions.
You MUST respond in pure JSON format as a list of dictionaries. Do not include any markdown backticks (like ```json), just the raw JSON array.
Each dictionary must have exactly these keys:
- "outfit": A simple description of the main clothing.
- "footwear": Suggested shoes.
- "grooming": Facial styling, grooming, or makeup suggestions.
- "accessories": Jewelry or other accessories.

Output pure JSON only."""
        gen_response = gen_model.generate_content(prompt)

        reply_text = gen_response.text.strip()
        if reply_text.startswith("```json"):
            reply_text = reply_text[7:-3].strip()
        elif reply_text.startswith("```"):
            reply_text = reply_text[3:-3].strip()

        try:
            suggestions = json.loads(reply_text)
            return jsonify({'type': 'suggestions', 'data': suggestions})
        except json.JSONDecodeError:
            return jsonify({'type': 'text', 'reply': reply_text})

    except Exception as e:
        logger.warning(f"Failed to generate chat response: {e}")
        return jsonify({'error': f"The stylist is unavailable right now. (Details: {str(e)})"}), 500


if __name__ == '__main__':
    # Ensure model is preloaded
    try:
        get_model()
    except Exception as e:
        logger.warning(f"Could not preload model: {e}")

    # Start camera manager (defaults to first available USB/Webcam device or mock)
    try:
        camera_manager.start()
    except Exception as e:
        logger.warning(f"Could not initialize camera manager on startup: {e}")

    app.run(debug=True, host='0.0.0.0', port=5000)
