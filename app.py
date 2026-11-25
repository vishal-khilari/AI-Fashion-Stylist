"""
app.py — Enhanced Flask Web Server for AI Fashion Stylist & Dress Classifier
-----------------------------------------------------------------------------
Features:
- Dual AI clothing inference engine: ONNX Runtime (fast, serverless) + PyTorch fallback
- Multi-person outfit detection & classification: Detects each individual person,
  crops their outfit, classifies it as Formal, Casual, or Traditional, and draws
  visually distinguishable colored bounding boxes (Formal=Blue, Casual=Green, Traditional=Orange).
- Real-time USB / Mobile Camera live streaming & automatic frame classification
- Localhost security guard for camera management endpoints
- Dynamic styling feedback powered by Google GenAI (gemini-2.5-flash)
- Interactive Fashion Chatbot API with structured JSON outfit recommendations
"""

import io
import os
import base64
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from PIL import Image, UnidentifiedImageError
from urllib.parse import urlparse

# Camera manager for USB / Mobile / WebRTC capture
from camera_manager import CameraManager, scan_available_cameras

# Gemini integration helper
import gemini_helper

# Multi-person detection & classification engine
import person_detector

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


# Class-specific metadata and color requirements:
# Formal: Blue (#2b7fff), Casual: Green (#2ecc71), Traditional: Orange (#ff8c00)
CLASS_META = {
    'Formal': {
        'emoji': '👔',
        'desc': 'Sophisticated attire: Suits, blazers, and elegant dresses.',
        'color': '#2b7fff',       # Blue
        'rgb': (43, 127, 255),
        'bgr': (255, 127, 43),
    },
    'Casual': {
        'emoji': '👕',
        'desc': 'Comfortable everyday wear: T-shirts, jeans, and hoodies.',
        'color': '#2ecc71',       # Green
        'rgb': (46, 204, 113),
        'bgr': (113, 204, 46),
    },
    'Traditional': {
        'emoji': '🥻',
        'desc': 'Cultural heritage: Ethnic wear like Sarees, Kurtas, and Sherwanis.',
        'color': '#ff8c00',       # Orange
        'rgb': (255, 140, 0),
        'bgr': (0, 140, 255),
    },
}

# Fallback metadata for any classes not explicitly defined above
DEFAULT_META = {
    'emoji': '👗',
    'desc': 'Clothing item detected.',
    'color': '#c9a96e',
    'rgb': (201, 169, 110),
    'bgr': (110, 169, 201),
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
                return True
        except Exception:
            return False
    return False


def run_inference_on_pil_image(img: Image.Image, generate_ai_feedback: bool = False) -> dict:
    """
    Core AI prediction pipeline for a single PIL Image (full frame or crop).
    Classifies clothing style into Formal, Casual, or Traditional.
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
            'score': round(float(probs_arr[i] * 100), 1),
            'emoji': meta['emoji'],
            'desc': meta['desc'],
            'color': meta['color']
        })

    all_scores.sort(key=lambda x: x['score'], reverse=True)
    top_meta = CLASS_META.get(top_class, DEFAULT_META)

    # 5. Optional Gemini feedback (only requested if needed, e.g. single-image inference)
    ai_feedback = None
    if generate_ai_feedback:
        ai_feedback = gemini_helper.generate_outfit_advice([
            {'person_label': 'Your outfit', 'predicted_class': top_class, 'confidence': round(top_prob * 100, 1)}
        ])

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


def run_multi_person_pipeline(
    img: Image.Image,
    generate_ai_feedback: bool = True,
    classify_attire: bool = True
) -> dict:
    """
    Run multi-person detection, crop each individual, conditionally classify their attire,
    and attach Gemini GenAI styling advice.
    """
    # Run multi-person detection + individual crop classification
    result = person_detector.process_multi_person_image(
        img,
        classifier_func=lambda crop: run_inference_on_pil_image(crop, generate_ai_feedback=False),
        classify_attire=classify_attire
    )

    # Generate unified AI styling feedback for all detected persons (only if attire classified)
    if generate_ai_feedback and classify_attire:
        try:
            advice = gemini_helper.generate_outfit_advice(result['persons'])
            result['ai_feedback'] = advice
        except Exception as e:
            logger.warning(f"Could not generate Gemini outfit advice: {e}")
            result['ai_feedback'] = "You look fantastic!"
    else:
        result['ai_feedback'] = None

    return result


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
        'camera': cam_status,
        'has_gemini_key': bool(gemini_helper.get_gemini_api_key())
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
    detect all persons, classify their attire, and return annotated predictions.
    """
    if not camera_manager.is_running:
        camera_manager.start()

    frame_pil = camera_manager.get_current_frame(as_pil=True)
    if frame_pil is None:
        status_info = camera_manager.get_status()
        err_msg = status_info.get('error_message') or f"No live frame available. Camera status: {status_info.get('status')}."
        return jsonify({'error': err_msg, 'status': status_info}), 400

    try:
        result = run_multi_person_pipeline(frame_pil, generate_ai_feedback=True)
        return jsonify(result)
    except Exception as e:
        logger.exception("Error during camera live prediction")
        return jsonify({'error': f"Inference failed: {str(e)}"}), 500


@app.route('/api/camera/detect_live', methods=['POST'])
def detect_live():
    """
    Fast live camera detection endpoint for real-time WebRTC or camera frame streams.
    Supports independent Person Detection and Attire Matching toggles.
    Uses ByteTrack / IoU tracking for stable bounding boxes across frames.
    """
    try:
        data = request.get_json(silent=True) or {}
        classify_attire = data.get('classify_attire', False)
        if isinstance(classify_attire, str):
            classify_attire = classify_attire.lower() in ('true', '1', 'yes', 'on')
        else:
            classify_attire = bool(classify_attire)

        frame_pil = None
        # Check if an image file was posted
        if 'image' in request.files:
            file = request.files['image']
            frame_pil = Image.open(io.BytesIO(file.read())).convert('RGB')
        else:
            # Check for base64 JSON payload
            b64_str = data.get('image')
            if b64_str:
                if ',' in b64_str:
                    b64_str = b64_str.split(',', 1)[1]
                img_bytes = base64.b64decode(b64_str)
                frame_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            else:
                # Grab latest frame from camera manager
                frame_pil = camera_manager.get_current_frame(as_pil=True)

        if frame_pil is None:
            return jsonify({'error': 'No image frame available for detection'}), 400

        # Run real-time multi-person tracking + conditional attire classification
        result = person_detector.track_and_process_live_frame(
            frame_pil,
            classifier_func=lambda crop: run_inference_on_pil_image(crop, generate_ai_feedback=False),
            classify_attire=classify_attire
        )
        return jsonify(result)

    except Exception as e:
        logger.exception("Error during live camera detection")
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# Standard Image Upload & Chat APIs
# ==============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload (multipart/form-data) with multi-person outfit detection.
    Detects each person in the frame, classifies their attire into Formal, Casual,
    or Traditional, and returns annotated image + per-person results.
    """
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

        result = run_multi_person_pipeline(img, generate_ai_feedback=True)
        return jsonify(result)

    except Exception as e:
        logger.exception("Error during multi-person file prediction")
        return jsonify({'error': f"An internal error occurred: {str(e)}"}), 500


@app.route('/chat_api', methods=['POST'])
def chat_api():
    """
    Handle chatbot messages using Gemini 2.5 Flash via google-genai.
    Conditionally returns 3-option outfit recommendations or conversational answers based on intent.
    """
    data = request.get_json(silent=True) or {}
    user_message = data.get('message')
    history = data.get('history', [])

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    result = gemini_helper.generate_chat_response(user_message, history=history)
    if not result.get('success'):
        return jsonify({'error': result.get('error', 'Failed to generate recommendations')}), 400

    if result.get('type') == 'suggestions':
        return jsonify({
            'type': 'suggestions',
            'data': result.get('data'),
            'reply': result.get('reply', 'Here are 3 complete looks for you: ✨')
        })
    else:
        return jsonify({
            'type': 'text',
            'reply': result.get('reply', '')
        })


if __name__ == '__main__':
    # Ensure classification model is preloaded
    try:
        get_model()
    except Exception as e:
        logger.warning(f"Could not preload dress classifier: {e}")

    # Ensure person detector is preloaded
    try:
        person_detector.get_detector()
    except Exception as e:
        logger.warning(f"Could not preload person detector: {e}")

    # Start camera manager (defaults to first available USB/Webcam device or mock)
    try:
        camera_manager.start()
    except Exception as e:
        logger.warning(f"Could not initialize camera manager on startup: {e}")

    app.run(debug=True, host='0.0.0.0', port=5000)
