"""
live_camera.py — Real-Time USB Mobile Camera Classifier & Desktop HUD
----------------------------------------------------------------------
Usage:
    python live_camera.py                          # Auto-detect best camera
    python live_camera.py --source 1               # USB Mobile Camera (Index 1)
    python live_camera.py --source http://127.0.0.1:8080/video   # IP Webcam over USB
    python live_camera.py --source mock            # Test with synthetic frames

Controls:
    [q] or [ESC] : Quit application
    [s]          : Request Gemini AI Fashion Stylist advice for current frame
    [c]          : Switch to next available camera source
    [SPACE]      : Freeze / Unfreeze display
"""

import cv2
import sys
import time
import argparse
import logging
import numpy as np
from PIL import Image
import os

from camera_manager import CameraManager, scan_available_cameras
from app import get_model, preprocess_image_numpy, CLASS_NAMES, CLASS_META, DEFAULT_META, HAS_ORT, HAS_TORCH, DEVICE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Mobile USB Camera Dress Classifier")
    parser.add_argument('--source', type=str, default=None,
                        help="Camera source: index (e.g. '0', '1'), stream URL, or 'mock'")
    parser.add_argument('--width', type=int, default=1280, help="Camera capture width")
    parser.add_argument('--height', type=int, default=720, help="Camera capture height")
    parser.add_argument('--fps', type=int, default=30, help="Target capture FPS")
    parser.add_argument('--interval', type=int, default=3, help="Run AI inference every N frames")
    parser.add_argument('--headless-frames', type=int, default=0,
                        help="Test mode: run for N frames without GUI window and exit")
    return parser.parse_args()


def request_gemini_styling(predicted_class: str, confidence: float):
    """Fetch styling advice from Gemini API in background/console."""
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key and os.path.exists("api_key.txt"):
        try:
            with open("api_key.txt", "r") as f:
                gemini_api_key = f.read().strip()
        except Exception:
            pass

    if not gemini_api_key:
        print("\n[AI Stylist] Set GEMINI_API_KEY in environment or api_key.txt to receive live advice.\n")
        return

    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        gen_model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = (
            f"Act as an expert fashion stylist. The user is on live camera wearing an outfit "
            f"our AI classified as {predicted_class} with {confidence:.1f}% confidence. "
            "Give a punchy, energetic 2-sentence fashion compliment and styling tip."
        )
        print("\n✦ Consulting AI Stylist...")
        response = gen_model.generate_content(prompt)
        print(f"\n✨ [AI Stylist Feedback]:\n{response.text}\n")
    except Exception as e:
        print(f"\n[AI Stylist Error]: {e}\n")


def hex_to_bgr(hex_color: str):
    """Convert hex color (e.g. '#4a90d9') to OpenCV BGR tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)
    return (201, 169, 110)


def draw_hud(frame: np.ndarray, prediction_data: dict, current_fps: float, source_name: str, frozen: bool):
    """Draw stylish real-time visual overlays on the camera frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top HUD bar
    hud_overlay = frame.copy()
    cv2.rectangle(hud_overlay, (0, 0), (w, 60), (14, 14, 14), -1)
    # Side panel for scores
    panel_w = 340
    cv2.rectangle(hud_overlay, (w - panel_w - 20, 70), (w - 20, 240), (18, 18, 18), -1)
    cv2.addWeighted(hud_overlay, 0.75, frame, 0.25, 0, frame)

    # Borders for panel
    cv2.rectangle(frame, (w - panel_w - 20, 70), (w - 20, 240), (45, 45, 45), 1)

    # Title & FPS
    cv2.putText(frame, "STYLESENSE - LIVE USB CAMERA", (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (201, 169, 110), 2, cv2.LINE_AA)
    
    fps_text = f"FPS: {current_fps:.1f} | SRC: {source_name}"
    if frozen:
        fps_text += " [FROZEN]"
    cv2.putText(frame, fps_text, (w - 380, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    # Center bounding guide
    box_w, box_h = int(w * 0.45), int(h * 0.7)
    bx1, by1 = (w - box_w) // 2, (h - box_h) // 2 + 20
    bx2, by2 = bx1 + box_w, by1 + box_h
    # Corner brackets
    corner_len = 30
    color = (201, 169, 110)
    # Top-Left
    cv2.line(frame, (bx1, by1), (bx1 + corner_len, by1), color, 2)
    cv2.line(frame, (bx1, by1), (bx1, by1 + corner_len), color, 2)
    # Top-Right
    cv2.line(frame, (bx2, by1), (bx2 - corner_len, by1), color, 2)
    cv2.line(frame, (bx2, by1), (bx2, by1 + corner_len), color, 2)
    # Bottom-Left
    cv2.line(frame, (bx1, by2), (bx1 + corner_len, by2), color, 2)
    cv2.line(frame, (bx1, by2), (bx1, by2 - corner_len), color, 2)
    # Bottom-Right
    cv2.line(frame, (bx2, by2), (bx2 - corner_len, by2), color, 2)
    cv2.line(frame, (bx2, by2), (bx2, by2 - corner_len), color, 2)

    # If prediction available, draw side score bars
    if prediction_data:
        top_cls = prediction_data.get('predicted_class', 'Detecting')
        top_conf = prediction_data.get('confidence', 0.0)
        scores = prediction_data.get('scores', [])

        # Top class header in panel
        meta = CLASS_META.get(top_cls, DEFAULT_META)
        top_color = hex_to_bgr(meta['color'])

        cv2.putText(frame, f"CLASS: {top_cls.upper()}", (w - panel_w, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, top_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Confidence: {top_conf:.1f}%", (w - panel_w, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw probability bars
        bar_y = 150
        bar_max_w = 180
        for s in scores:
            c_name = s['class']
            pct = s['score']
            bar_color = hex_to_bgr(s['color'])
            fill_w = int((pct / 100.0) * bar_max_w)

            cv2.putText(frame, f"{c_name:<11}", (w - panel_w, bar_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

            # Bar background
            cv2.rectangle(frame, (w - panel_w + 100, bar_y), (w - panel_w + 100 + bar_max_w, bar_y + 14), (35, 35, 35), -1)
            # Bar fill
            if fill_w > 0:
                cv2.rectangle(frame, (w - panel_w + 100, bar_y), (w - panel_w + 100 + fill_w, bar_y + 14), bar_color, -1)

            cv2.putText(frame, f"{pct:.0f}%", (w - panel_w + 100 + bar_max_w + 10, bar_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (201, 169, 110), 1, cv2.LINE_AA)

            bar_y += 26

    # Bottom helper bar
    cv2.putText(frame, "[Q] Quit  |  [S] AI Stylist Advice  |  [C] Switch Camera  |  [SPACE] Freeze",
                (24, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 140, 140), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    print("=========================================================")
    print("   StyleSense — Live Real-Time Mobile Camera Classifier  ")
    print("=========================================================")

    # Initialize model
    print("Loading AI Model...")
    model_obj = get_model()
    print("Model ready!")

    # Setup camera manager
    cam = CameraManager(default_source=args.source, width=args.width, height=args.height, fps=args.fps)
    cam.start()

    window_name = "StyleSense - Real-Time Mobile USB Camera"
    if args.headless_frames == 0:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)

    frame_idx = 0
    latest_prediction = None
    frozen = False
    frozen_frame = None

    try:
        while True:
            if not frozen:
                frame = cam.get_current_frame(as_pil=False)
                if frame is None:
                    time.sleep(0.02)
                    continue

                frame_idx += 1
                # Run inference periodically
                if frame_idx % args.interval == 0:
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if HAS_ORT:
                        numpy_tensor = preprocess_image_numpy(pil_img)
                        input_name = model_obj.get_inputs()[0].name
                        logits = model_obj.run(None, {input_name: numpy_tensor})[0]
                        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                        probs_arr = (e_x / e_x.sum(axis=1, keepdims=True))[0]
                        top_idx = int(np.argmax(probs_arr))
                        top_prob = float(probs_arr[top_idx])
                    else:
                        import torch
                        from Transforms import inference_transform
                        tensor = inference_transform(pil_img).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            logits = model_obj(tensor)
                            probs = torch.softmax(logits, dim=1)[0]
                        probs_arr = probs.cpu().numpy()
                        top_idx = probs.argmax().item()
                        top_prob = probs[top_idx].item()

                    scores = []
                    for i, c in enumerate(CLASS_NAMES):
                        meta = CLASS_META.get(c, DEFAULT_META)
                        scores.append({
                            'class': c,
                            'score': float(probs_arr[i] * 100),
                            'color': meta['color']
                        })
                    scores.sort(key=lambda x: x['score'], reverse=True)

                    latest_prediction = {
                        'predicted_class': CLASS_NAMES[top_idx],
                        'confidence': round(top_prob * 100, 1),
                        'scores': scores
                    }

                display_frame = frame.copy()
            else:
                display_frame = frozen_frame.copy()

            # Render HUD
            status = cam.get_status()
            draw_hud(display_frame, latest_prediction, status['current_fps'], status['source'], frozen)

            if args.headless_frames > 0:
                if frame_idx >= args.headless_frames:
                    print(f"Headless test complete: processed {frame_idx} frames successfully.")
                    break
                time.sleep(0.01)
                continue

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):  # 'q' or ESC
                break
            elif key == ord(' '):  # SPACE to freeze/unfreeze
                frozen = not frozen
                if frozen:
                    frozen_frame = frame.copy()
            elif key == ord('s'):  # 's' for AI Stylist feedback
                if latest_prediction:
                    request_gemini_styling(latest_prediction['predicted_class'], latest_prediction['confidence'])
            elif key == ord('c'):  # 'c' to switch camera
                devs = scan_available_cameras()
                dev_ids = [d['id'] for d in devs]
                if dev_ids:
                    cur_idx = dev_ids.index(str(cam.source)) if str(cam.source) in dev_ids else 0
                    next_id = dev_ids[(cur_idx + 1) % len(dev_ids)]
                    print(f"\n[Camera Switch] Switching source to: {next_id}")
                    cam.switch_source(next_id)

    except KeyboardInterrupt:
        pass
    finally:
        print("\nCleaning up resources...")
        cam.release()
        if args.headless_frames == 0:
            cv2.destroyAllWindows()
        print("Done.")


if __name__ == '__main__':
    main()
