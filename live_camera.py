"""
live_camera.py — Real-Time USB Mobile Camera Classifier & Desktop HUD
----------------------------------------------------------------------
Multi-Person Outfit Detection Edition:
  - Detects and classifies all people appearing in the camera frame.
  - Draws colored bounding boxes:
      * Formal: Blue
      * Casual: Green
      * Traditional: Orange
  - Displays Person Number, Attire Category, and Clothing Match % on each box.
  - Dynamic AI Fashion Stylist advice for multiple people powered by gemini-2.5-flash.

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
import gemini_helper
import person_detector

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
    parser.add_argument('--detect-person', action='store_true', default=False,
                        help="Start with Person Detection enabled (default: OFF)")
    parser.add_argument('--match-attire', action='store_true', default=False,
                        help="Start with Attire Matching enabled (default: OFF)")
    parser.add_argument('--headless-frames', type=int, default=0,
                        help="Test mode: run for N frames without GUI window and exit")
    return parser.parse_args()


def request_gemini_styling(multi_person_result: dict):
    """Fetch styling advice from Gemini API for all detected people."""
    persons = multi_person_result.get('persons', [])
    if not persons:
        print("\n[AI Stylist] No persons currently detected in frame.\n")
        return

    print(f"\n✦ Consulting Gemini AI Stylist for {len(persons)} person(s)...")
    advice = gemini_helper.generate_outfit_advice(persons)
    print(f"\n✨ [AI Stylist Feedback (gemini-2.5-flash)]:\n{advice}\n")


def hex_to_bgr(hex_color: str):
    """Convert hex color (e.g. '#2b7fff') to OpenCV BGR tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)
    return (255, 127, 43)


def draw_hud(
    frame: np.ndarray,
    multi_person_result: dict,
    current_fps: float,
    source_name: str,
    frozen: bool,
    detect_person: bool = False,
    match_attire: bool = False
):
    """Draw stylish real-time visual overlays and side HUD on the camera frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top HUD bar
    hud_overlay = frame.copy()
    cv2.rectangle(hud_overlay, (0, 0), (w, 55), (14, 14, 14), -1)

    # Side panel for multi-person scores
    persons = multi_person_result.get('persons', []) if (multi_person_result and detect_person) else []
    panel_w = 340
    panel_h = max(180, min(500, 70 + len(persons) * 75))
    cv2.rectangle(hud_overlay, (w - panel_w - 20, 65), (w - 20, 65 + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(hud_overlay, 0.75, frame, 0.25, 0, frame)

    # Borders for panel
    cv2.rectangle(frame, (w - panel_w - 20, 65), (w - 20, 65 + panel_h), (45, 45, 45), 1)

    # Title & FPS
    cv2.putText(frame, "STYLESENSE — MULTI-PERSON LIVE CAMERA", (24, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (201, 169, 110), 2, cv2.LINE_AA)

    fps_text = f"FPS: {current_fps:.1f} | SRC: {source_name}"
    if frozen:
        fps_text += " [FROZEN]"
    cv2.putText(frame, fps_text, (w - 380, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    # Draw detected person boxes onto frame if detection is ON
    if detect_person and persons:
        for p in persons:
            x1, y1, x2, y2 = p['box']
            cls = p.get('predicted_class')
            conf = p.get('confidence', 0.0)

            if match_attire and cls:
                meta = CLASS_META.get(cls, DEFAULT_META)
                color_bgr = meta.get('bgr', hex_to_bgr(meta['color']))
                label_text = f"{p['person_label']}: {cls} ({conf:.1f}%)"
                if p.get('is_low_match'):
                    label_text += " [Low Match]"
            else:
                # Neutral bounding box color when Attire Matching is OFF
                color_bgr = (110, 169, 201)  # Neutral accent gold (BGR)
                label_text = f"{p['person_label']}"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 3)

            # Label badge
            font_scale = 0.55
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            badge_y1 = y1 - th - 10
            if badge_y1 < 0:
                badge_y1 = y1 + 4
            badge_y2 = badge_y1 + th + 10
            badge_x1 = max(0, min(w - tw - 12, x1))
            badge_x2 = badge_x1 + tw + 12

            cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color_bgr, -1)
            cv2.putText(frame, label_text, (badge_x1 + 6, badge_y2 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Side Panel Content: Toggle Status & Detected People List
    detect_status_str = "ON" if detect_person else "OFF"
    attire_status_str = "ON" if match_attire else "OFF"

    cv2.putText(frame, f"DETECTION: {detect_status_str} | ATTIRE: {attire_status_str}",
                (w - panel_w, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (201, 169, 110), 1, cv2.LINE_AA)

    cur_y = 125
    if not detect_person:
        cv2.putText(frame, "Person Detection is OFF.", (w - panel_w, cur_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(frame, "Press [D] to enable boxes.", (w - panel_w, cur_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)
    elif not persons:
        cv2.putText(frame, "Scanning for people in frame...", (w - panel_w, cur_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 140, 140), 1, cv2.LINE_AA)
    else:
        for p in persons:
            cls = p.get('predicted_class')
            conf = p.get('confidence', 0.0)

            if match_attire and cls:
                meta = CLASS_META.get(cls, DEFAULT_META)
                color_bgr = meta.get('bgr', hex_to_bgr(meta['color']))
                header_text = f"{p['person_label']} : {cls.upper()}"
                pct_text = f"{conf:.1f}% Match"
            else:
                color_bgr = (110, 169, 201)
                header_text = f"{p['person_label']}"
                pct_text = "Detected"

            # Header for person
            cv2.putText(frame, header_text, (w - panel_w, cur_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2, cv2.LINE_AA)
            cv2.putText(frame, pct_text, (w - 110, cur_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

            # Mini score bar for top class if attire matching active
            if match_attire and conf > 0:
                bar_w = int((conf / 100.0) * (panel_w - 40))
                cv2.rectangle(frame, (w - panel_w, cur_y + 8), (w - 40, cur_y + 16), (35, 35, 35), -1)
                if bar_w > 0:
                    cv2.rectangle(frame, (w - panel_w, cur_y + 8), (w - panel_w + bar_w, cur_y + 16), color_bgr, -1)

            cur_y += 55

    # Bottom helper bar with 2 independent toggle keys
    cv2.putText(frame, "[D] Person Boxes [OFF/ON] | [A] Attire Match [OFF/ON] | [S] Stylist | [Q] Quit",
                (24, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)


def classify_crop_local(crop_pil: Image.Image, model_obj) -> dict:
    """Classify a single crop locally using ONNX or PyTorch."""
    if HAS_ORT:
        numpy_tensor = preprocess_image_numpy(crop_pil)
        input_name = model_obj.get_inputs()[0].name
        logits = model_obj.run(None, {input_name: numpy_tensor})[0]
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs_arr = (e_x / e_x.sum(axis=1, keepdims=True))[0]
        top_idx = int(np.argmax(probs_arr))
        top_prob = float(probs_arr[top_idx])
    else:
        import torch
        from Transforms import inference_transform
        tensor = inference_transform(crop_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model_obj(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        probs_arr = probs.cpu().numpy()
        top_idx = probs.argmax().item()
        top_prob = probs[top_idx].item()

    top_class = CLASS_NAMES[top_idx]
    scores = []
    for i, c in enumerate(CLASS_NAMES):
        meta = CLASS_META.get(c, DEFAULT_META)
        scores.append({
            'class': c,
            'score': round(float(probs_arr[i] * 100), 1),
            'color': meta['color']
        })
    scores.sort(key=lambda x: x['score'], reverse=True)

    return {
        'predicted_class': top_class,
        'confidence': round(top_prob * 100, 1),
        'scores': scores
    }


def main():
    args = parse_args()
    print("=========================================================")
    print("   StyleSense — Multi-Person Live Camera Dress Classifier ")
    print("=========================================================")

    # Initialize model
    print("Loading AI Clothing Classifier...")
    model_obj = get_model()
    print("Clothing classifier ready!")

    # Initialize person detector
    print("Loading Person Detector...")
    person_detector.get_detector()
    print("Person detector ready!")

    # Setup camera manager
    cam = CameraManager(default_source=args.source, width=args.width, height=args.height, fps=args.fps)
    cam.start()

    window_name = "StyleSense - Multi-Person Dress Classifier"
    if args.headless_frames == 0:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)

    frame_idx = 0
    latest_multi_result = None
    frozen = False
    frozen_frame = None

    # Two independent toggles (Default: OFF)
    detect_person = bool(args.detect_person)
    match_attire = bool(args.match_attire)
    if match_attire:
        detect_person = True  # Attire matching requires person detection

    try:
        while True:
            if not frozen:
                frame = cam.get_current_frame(as_pil=False)
                if frame is None:
                    time.sleep(0.02)
                    continue

                frame_idx += 1
                # Run real-time tracked multi-person inference periodically when detection is ON
                if detect_person and (frame_idx % args.interval == 0):
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    latest_multi_result = person_detector.track_and_process_live_frame(
                        pil_img,
                        classifier_func=lambda crop: classify_crop_local(crop, model_obj),
                        classify_attire=match_attire
                    )
                elif not detect_person:
                    latest_multi_result = None

                display_frame = frame.copy()
            else:
                display_frame = frozen_frame.copy()

            # Render HUD and Bounding Boxes according to toggle states
            status = cam.get_status()
            draw_hud(
                display_frame,
                latest_multi_result,
                status['current_fps'],
                status['source'],
                frozen,
                detect_person=detect_person,
                match_attire=match_attire
            )

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
            elif key in (ord('d'), ord('D')):
                detect_person = not detect_person
                if not detect_person:
                    match_attire = False
                    latest_multi_result = None
                print(f"[Toggle] Person Detection: {'ON' if detect_person else 'OFF'} | Attire Matching: {'ON' if match_attire else 'OFF'}")
            elif key in (ord('a'), ord('A')):
                match_attire = not match_attire
                if match_attire:
                    detect_person = True  # Enabling Attire Matching automatically enables Person Detection
                print(f"[Toggle] Attire Matching: {'ON' if match_attire else 'OFF'} | Person Detection: {'ON' if detect_person else 'OFF'}")
            elif key == ord(' '):  # SPACE to freeze/unfreeze
                frozen = not frozen
                if frozen:
                    frozen_frame = frame.copy()
            elif key in (ord('s'), ord('S')):  # 's' for AI Stylist feedback
                if latest_multi_result and detect_person:
                    request_gemini_styling(latest_multi_result)
                else:
                    print("\n[AI Stylist] Enable Person Detection [D] first to detect people for styling advice.\n")
            elif key in (ord('c'), ord('C')):  # 'c' to switch camera
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
