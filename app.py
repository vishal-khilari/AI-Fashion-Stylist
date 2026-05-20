"""
app.py — Enhanced Flask web server for the Dress Classifier
-----------------------------------------------------------
Features:
- Robust error handling for image processing
- Detailed class metadata with modern styling
- Dynamic response structure for the frontend
"""

import io
import os
import torch
import logging
from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError

# Project-specific imports
from initialisation import CLASS_NAMES, DEVICE
from Transforms import inference_transform
from SaveAndReload import load_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Model configuration
MODEL_PATH = 'dress_classifier.pth'
model = None

def get_model():
    """Lazy load the model to ensure it's ready when needed."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")
        
        logger.info("Loading dress classifier model...")
        model = load_model(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        logger.info(f"Model loaded successfully on {DEVICE}!")
    return model

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

@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('home.html')

@app.route('/upload/<category>')
def upload(category):
    """Serve the image upload page for a specific category."""
    return render_template('upload.html', category=category)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return style predictions."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # 1. Load and validate image
        img_bytes = file.read()
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except UnidentifiedImageError:
            return jsonify({'error': 'Invalid image format. Please upload a JPG, PNG, or WEBP.'}), 400

        # 2. Preprocess for the model
        tensor = inference_transform(img).unsqueeze(0).to(DEVICE)

        # 3. Inference
        current_model = get_model()
        with torch.no_grad():
            logits = current_model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        # 4. Process results
        top_idx = probs.argmax().item()
        top_class = CLASS_NAMES[top_idx]
        
        all_scores = []
        for i, class_name in enumerate(CLASS_NAMES):
            meta = CLASS_META.get(class_name, DEFAULT_META)
            all_scores.append({
                'class': class_name,
                'score': float(probs[i].item() * 100),
                'emoji': meta['emoji'],
                'desc': meta['desc'],
                'color': meta['color']
            })

        # Sort scores by highest probability
        all_scores.sort(key=lambda x: x['score'], reverse=True)

        top_meta = CLASS_META.get(top_class, DEFAULT_META)
        
        logger.info(f"Prediction: {top_class} ({probs[top_idx].item():.2f})")

        # --- GenAI Integration ---
        ai_feedback = None
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key and os.path.exists("api_key.txt"):
            with open("api_key.txt", "r") as f:
                gemini_api_key = f.read().strip()

        if gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
                gen_model = genai.GenerativeModel('gemini-2.5-flash')
                prompt = f"Act as an expert fashion stylist. The user has uploaded an outfit that our AI classified as {top_class} with {round(probs[top_idx].item() * 100, 1)}% confidence. Give a short, energetic 2-sentence fashion compliment or styling tip. Be creative and hype them up!"
                gen_response = gen_model.generate_content(prompt)
                ai_feedback = gen_response.text
            except Exception as e:
                logger.warning(f"Failed to generate AI feedback: {e}")
                ai_feedback = "You look fantastic! (GenAI stylist is currently unavailable due to an error)."
        else:
            ai_feedback = "Set your GEMINI_API_KEY environment variable to get personalized GenAI styling advice!"
        # -------------------------

        return jsonify({
            'success': True,
            'predicted_class': top_class,
            'confidence': round(float(probs[top_idx].item() * 100), 1),
            'emoji': top_meta['emoji'],
            'desc': top_meta['desc'],
            'color': top_meta['color'],
            'scores': all_scores,
            'ai_feedback': ai_feedback
        })

    except Exception as e:
        logger.exception("Error during prediction")
        return jsonify({'error': f"An internal error occurred: {str(e)}"}), 500

@app.route('/health')
def health():
    """Simple health check endpoint."""
    return jsonify({'status': 'healthy', 'device': str(DEVICE)})

@app.route('/chatbot')
def chatbot():
    """Serve the chatbot interface."""
    return render_template('chatbot.html')

@app.route('/chat_api', methods=['POST'])
def chat_api():
    """Handle chatbot messages using Gemini."""
    data = request.get_json()
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    gemini_api_key = None
    if os.path.exists("api_key.txt"):
        with open("api_key.txt", "r") as f:
            gemini_api_key = f.read().strip()
    if not gemini_api_key:
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
            
    if not gemini_api_key:
        return jsonify({'reply': "Please set your GEMINI_API_KEY environment variable or create an api_key.txt file to chat with the stylist!"})

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
    # Ensure the model exists before starting (or it will lazy load on first request)
    try:
        get_model()
    except Exception as e:
        logger.warning(f"Could not preload model: {e}")

    app.run(debug=True, host='0.0.0.0', port=5000)
