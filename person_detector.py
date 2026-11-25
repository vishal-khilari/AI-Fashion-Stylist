"""
person_detector.py — Multi-Person Outfit Detection & Classification Engine
--------------------------------------------------------------------------
Features:
  1. Detects all people in a camera frame or uploaded image using YOLOv8 (with fallback to SSDLite).
  2. Crops each individual person separately with adaptive padding.
  3. Classifies each person's outfit using the existing dress_classifier (ONNX / PyTorch)
     into Formal, Casual, or Traditional.
  4. Preserves the exact trained class mapping: 0=Formal, 1=Casual, 2=Traditional.
  5. Computes clothing classifier match percentage for each individual person.
  6. Visual annotation with distinguishable category colors:
       - Formal: Blue (#2b7fff)
       - Casual: Green (#2ecc71)
       - Traditional: Orange (#ff8c00)
  7. Renders clear, non-overlapping bounding boxes and labels:
       "Person 1: Formal (94.5%)"
  8. Handles zero-person fallback (analyzes whole frame) and flags low-match predictions (<45%).
"""

import os
import io
import time
import base64
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Category mapping and visual color specifications
CATEGORY_META = {
    'Formal': {
        'hex': '#2b7fff',          # Blue
        'rgb': (43, 127, 255),
        'bgr': (255, 127, 43),     # OpenCV BGR
        'emoji': '👔',
        'desc': 'Sophisticated attire: Suits, blazers, and elegant dresses.'
    },
    'Casual': {
        'hex': '#2ecc71',          # Green
        'rgb': (46, 204, 113),
        'bgr': (113, 204, 46),     # OpenCV BGR
        'emoji': '👕',
        'desc': 'Comfortable everyday wear: T-shirts, jeans, and hoodies.'
    },
    'Traditional': {
        'hex': '#ff8c00',          # Orange
        'rgb': (255, 140, 0),
        'bgr': (0, 140, 255),      # OpenCV BGR
        'emoji': '🥻',
        'desc': 'Cultural heritage: Ethnic wear like Sarees, Kurtas, and Sherwanis.'
    }
}

DEFAULT_META = {
    'hex': '#c9a96e',
    'rgb': (201, 169, 110),
    'bgr': (110, 169, 201),
    'emoji': '👗',
    'desc': 'Clothing item detected.'
}

NEUTRAL_BOX_COLOR = DEFAULT_META['hex']

CLASS_NAMES = ['Formal', 'Casual', 'Traditional']

# Global singletons for models
_detector_model = None
_detector_type = None  # 'yolo' or 'ssdlite' or 'hog'


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def apply_nms(boxes_with_conf: List[Tuple[List[int], float]], iou_thresh: float = 0.5) -> List[List[int]]:
    """Apply Non-Maximum Suppression to remove duplicate person detections."""
    if not boxes_with_conf:
        return []

    # Sort descending by detection confidence
    sorted_boxes = sorted(boxes_with_conf, key=lambda x: x[1], reverse=True)
    kept = []

    for box, conf in sorted_boxes:
        overlap = False
        for k_box in kept:
            if compute_iou(box, k_box) > iou_thresh:
                overlap = True
                break
        if not overlap:
            kept.append(box)

    return kept


def get_detector():
    """
    Load person detection model once and reuse.
    Prefers YOLOv8n (ultra-fast, ~15-30ms), with fallback to SSDLite.
    """
    global _detector_model, _detector_type
    if _detector_model is not None:
        return _detector_model, _detector_type

    # 1. Try YOLOv8n
    try:
        from ultralytics import YOLO
        logger.info("Loading YOLOv8n person detector...")
        model = YOLO('yolov8n.pt')
        _detector_model = model
        _detector_type = 'yolo'
        logger.info("YOLOv8n person detector loaded successfully!")
        return _detector_model, _detector_type
    except Exception as e:
        logger.warning(f"Could not load YOLOv8n ({e}). Trying torchvision SSDLite...")

    # 2. Try torchvision SSDLite MobileNetV3
    try:
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
        logger.info("Loading Torchvision SSDLite MobileNetV3 detector...")
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        model = ssdlite320_mobilenet_v3_large(weights=weights).eval()
        _detector_model = model
        _detector_type = 'ssdlite'
        logger.info("Torchvision SSDLite detector loaded successfully!")
        return _detector_model, _detector_type
    except Exception as e:
        logger.warning(f"Could not load torchvision SSDLite ({e}). Using OpenCV HOG fallback...")

    # 3. Fallback: OpenCV HOG
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
        _detector_model = hog
        _detector_type = 'hog'
        logger.info("OpenCV HOG people detector loaded as fallback.")
        return _detector_model, _detector_type
    except Exception as e:
        logger.error(f"Failed to load any person detector: {e}")
        return None, None


def detect_person_boxes(img_rgb: np.ndarray, min_conf: float = 0.35) -> List[List[int]]:
    """
    Detect all people in an RGB image.
    Returns list of [x1, y1, x2, y2] bounding boxes, sorted from left to right.
    """
    h, w = img_rgb.shape[:2]
    detector, det_type = get_detector()
    if detector is None:
        return []

    raw_boxes = []

    if det_type == 'yolo':
        # Class 0 in COCO is person
        results = detector(img_rgb, classes=[0], conf=min_conf, verbose=False)[0]
        for box in results.boxes:
            conf = float(box.conf[0].item())
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(round(c)) for c in coords]
            # Clamp to image bounds
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            # Filter tiny slivers
            if (x2 - x1) >= 20 and (y2 - y1) >= 40:
                raw_boxes.append(([x1, y1, x2, y2], conf))

    elif det_type == 'ssdlite':
        import torch
        import torchvision.transforms.functional as F
        pil_img = Image.fromarray(img_rgb)
        tensor = F.to_tensor(pil_img).unsqueeze(0)
        with torch.no_grad():
            out = detector(tensor)[0]
        boxes = out['boxes']
        labels = out['labels']
        scores = out['scores']
        # In COCO label 1 is person
        for b, l, s in zip(boxes, labels, scores):
            if l.item() == 1 and s.item() >= min_conf:
                x1, y1, x2, y2 = [int(round(c)) for c in b.tolist()]
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                if (x2 - x1) >= 20 and (y2 - y1) >= 40:
                    raw_boxes.append(([x1, y1, x2, y2], float(s.item())))

    elif det_type == 'hog':
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        rects, weights = detector.detectMultiScale(bgr, winStride=(8, 8), padding=(4, 4), scale=1.05)
        for (rx, ry, rw, rh), weight in zip(rects, weights):
            conf = float(weight[0]) if hasattr(weight, '__len__') else float(weight)
            x1, y1, x2, y2 = rx, ry, rx + rw, ry + rh
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if (x2 - x1) >= 20 and (y2 - y1) >= 40:
                raw_boxes.append(([x1, y1, x2, y2], conf))

    # Apply NMS to remove overlapping duplicate boxes
    filtered_boxes = apply_nms(raw_boxes, iou_thresh=0.45)

    # Sort left to right so Person 1 is on the left, Person 2 is next, etc.
    filtered_boxes.sort(key=lambda b: b[0])
    return filtered_boxes


def expand_box_with_padding(box: List[int], img_w: int, img_h: int, pad_pct: float = 0.05) -> List[int]:
    """Slightly expand bounding box so full clothing (collars/shoes) is captured."""
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * pad_pct)
    py = int(bh * pad_pct)

    nx1 = max(0, x1 - px)
    ny1 = max(0, y1 - py)
    nx2 = min(img_w, x2 + px)
    ny2 = min(img_h, y2 + py)
    return [nx1, ny1, nx2, ny2]


def classify_single_crop(crop_pil: Image.Image, classifier_func) -> Tuple[str, float, List[Dict[str, Any]]]:
    """
    Run the clothing classifier on a single cropped person image.
    Returns (top_class, top_confidence_pct, all_scores_list).
    """
    pred_dict = classifier_func(crop_pil)
    top_class = pred_dict.get('predicted_class', 'Casual')
    top_conf = float(pred_dict.get('confidence', 0.0))
    scores = pred_dict.get('scores', [])

    # Enrich scores with updated category colors
    enriched_scores = []
    for s in scores:
        c_name = s.get('class', '')
        meta = CATEGORY_META.get(c_name, DEFAULT_META)
        enriched_scores.append({
            'class': c_name,
            'score': round(float(s.get('score', 0.0)), 1),
            'emoji': meta['emoji'],
            'desc': meta['desc'],
            'color': meta['hex']
        })

    return top_class, top_conf, enriched_scores


def annotate_image_with_detections(
    img_pil: Image.Image,
    persons: List[Dict[str, Any]],
    box_thickness: int = 3,
    classify_attire: bool = True
) -> Image.Image:
    """
    Draw visually distinguishable bounding boxes and informative labels on the image:
      - If classify_attire is False: neutral bounding boxes with "Person {id}" labels.
      - If classify_attire is True: category-colored boxes (Formal=Blue, Casual=Green, Traditional=Orange)
        with "Person {id}: {Class} ({Match}%)" labels.
    """
    annotated = img_pil.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    # Load font or fall back to default
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, int(h * 0.024)))
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    for p in persons:
        box = p['box']
        x1, y1, x2, y2 = box
        cls = p.get('predicted_class')
        conf = p.get('confidence', 0.0)

        if classify_attire and cls:
            meta = CATEGORY_META.get(cls, DEFAULT_META)
            color_rgb = meta['rgb']
            label_text = f"{p['person_label']}: {cls} ({conf:.1f}%)"
            if p.get('is_low_match'):
                label_text += " [Low Match]"
        else:
            # Neutral bounding box color when Attire Matching is OFF
            color_rgb = DEFAULT_META['rgb']
            label_text = f"{p['person_label']}"

        # 1. Draw Bounding Box Rectangle
        for offset in range(box_thickness):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color_rgb
            )

        # 2. Compute Label Badge Dimensions
        if font and hasattr(font, 'getbbox'):
            bbox = font.getbbox(label_text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w = len(label_text) * 8
            text_h = 16

        badge_pad_x = 8
        badge_pad_y = 5
        badge_w = text_w + badge_pad_x * 2
        badge_h = text_h + badge_pad_y * 2

        # 3. Position Label Badge (avoid going outside image top or right)
        badge_x1 = max(0, min(w - badge_w, x1))
        badge_y1 = y1 - badge_h
        if badge_y1 < 0:
            badge_y1 = y1 + 4  # Draw inside top of box if at upper boundary

        badge_x2 = badge_x1 + badge_w
        badge_y2 = badge_y1 + badge_h

        # Draw filled category-colored badge
        draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=color_rgb)

        # Draw white bold text
        text_x = badge_x1 + badge_pad_x
        text_y = badge_y1 + badge_pad_y
        draw.text((text_x, text_y), label_text, fill=(255, 255, 255), font=font)

    return annotated


def annotate_cv2_frame(
    frame_bgr: np.ndarray,
    persons: List[Dict[str, Any]],
    classify_attire: bool = True
) -> np.ndarray:
    """Fast in-place OpenCV drawing for live video streams."""
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    for p in persons:
        x1, y1, x2, y2 = p['box']
        cls = p.get('predicted_class')
        conf = p.get('confidence', 0.0)

        if classify_attire and cls:
            meta = CATEGORY_META.get(cls, DEFAULT_META)
            color_bgr = meta['bgr']
            text = f"{p['person_label']}: {cls} ({conf:.1f}%)"
            if p.get('is_low_match'):
                text += " [Low Match]"
        else:
            # Neutral color when Attire Matching is OFF
            color_bgr = DEFAULT_META['bgr']
            text = f"{p['person_label']}"

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 3)

        font_scale = max(0.5, min(0.8, w / 1200.0))
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        lx1 = max(0, min(w - tw - 12, x1))
        ly1 = y1 - th - 10
        if ly1 < 0:
            ly1 = y1 + 4

        lx2 = lx1 + tw + 12
        ly2 = ly1 + th + 10

        # Badge background
        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), color_bgr, -1)
        # White text
        cv2.putText(
            out, text, (lx1 + 6, ly2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
            thickness, cv2.LINE_AA
        )

    return out


def pil_to_base64_jpeg(pil_img: Image.Image, quality: int = 85) -> str:
    """Convert PIL image to base64 JPEG data URL."""
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"


# ==============================================================================
# Real-Time Multi-Person Tracking Engine (ByteTrack / IoU + EMA Smoothing)
# ==============================================================================

class PersonTracker:
    """
    Lightweight IoU + Centroid multi-person tracker with coordinate smoothing.
    Assigns persistent IDs to individuals across video frames, tracks people as
    they move, automatically registers people entering the frame, cleans up people
    who leave, and caches clothing classifications per tracking ID.
    """
    def __init__(self, max_disappeared: int = 10, iou_threshold: float = 0.3, ema_alpha: float = 0.70):
        self.next_id = 1
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.ema_alpha = ema_alpha
        # track_id -> dict of tracking state
        self.tracks: Dict[int, Dict[str, Any]] = {}

    @property
    def next_person_id(self) -> int:
        return self.next_id

    def reset(self):
        self.next_id = 1
        self.tracks.clear()

    def get_cached_classification(self, track_id: int, max_age_seconds: float = 1.0) -> Optional[Dict[str, Any]]:
        t = self.tracks.get(track_id)
        if not t or t.get('predicted_class') is None:
            return None
        if (time.time() - t.get('last_classified_time', 0.0)) > max_age_seconds:
            return None
        return {
            'predicted_class': t.get('predicted_class'),
            'confidence': t.get('confidence', 0.0),
            'scores': t.get('scores', [])
        }

    def set_cached_classification(self, track_id: int, classification_data: Dict[str, Any]):
        t = self.tracks.get(track_id)
        if t:
            t['predicted_class'] = classification_data.get('predicted_class')
            t['confidence'] = classification_data.get('confidence', 0.0)
            t['scores'] = classification_data.get('scores', [])
            t['last_classified_time'] = time.time()

    def update(
        self,
        new_boxes: List[List[int]],
        img_w: int = 640,
        img_h: int = 480,
        img_pil: Optional[Image.Image] = None,
        classifier_func=None,
        classify_attire: bool = True,
        min_clothing_conf_threshold: float = 45.0
    ) -> List[Dict[str, Any]]:
        if img_pil is None:
            img_pil = Image.new('RGB', (img_w, img_h))

        # 1. If no existing tracks, initialize all new boxes
        if not self.tracks:
            for box in new_boxes:
                self._create_track(box, img_w, img_h, img_pil, classifier_func, classify_attire, min_clothing_conf_threshold)
            return self._format_output(img_w, img_h, classify_attire)

        # 2. If new_boxes is empty, mark all existing tracks as disappeared
        if not new_boxes:
            dead_ids = []
            for tid, t in self.tracks.items():
                t['disappeared'] += 1
                if t['disappeared'] > self.max_disappeared:
                    dead_ids.append(tid)
            for tid in dead_ids:
                del self.tracks[tid]
            return []

        # 3. Match existing active tracks with new_boxes using IoU
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(new_boxes)), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            t_box = self.tracks[tid]['box']
            for j, n_box in enumerate(new_boxes):
                iou_matrix[i, j] = compute_iou(t_box, n_box)

        matched_tracks = set()
        matched_boxes = set()

        # Greedy matching from highest IoU down to iou_threshold
        while True:
            max_iou = 0.0
            best_i, best_j = -1, -1
            for i in range(len(track_ids)):
                if i in matched_tracks:
                    continue
                for j in range(len(new_boxes)):
                    if j in matched_boxes:
                        continue
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        best_i, best_j = i, j

            if max_iou < self.iou_threshold or best_i == -1:
                break

            matched_tracks.add(best_i)
            matched_boxes.add(best_j)

            tid = track_ids[best_i]
            n_box = new_boxes[best_j]
            self._update_track(tid, n_box, img_w, img_h, img_pil, classifier_func, classify_attire, min_clothing_conf_threshold)

        # 4. Handle unmatched existing tracks
        dead_ids = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid]['disappeared'] += 1
                if self.tracks[tid]['disappeared'] > self.max_disappeared:
                    dead_ids.append(tid)
        for tid in dead_ids:
            del self.tracks[tid]

        # 5. Handle unmatched new boxes (new people entering frame)
        for j, n_box in enumerate(new_boxes):
            if j not in matched_boxes:
                self._create_track(n_box, img_w, img_h, img_pil, classifier_func, classify_attire, min_clothing_conf_threshold)

        return self._format_output(img_w, img_h, classify_attire)

    def _create_track(self, box, img_w, img_h, img_pil, classifier_func, classify_attire, min_conf_thresh):
        tid = self._get_next_available_id()
        track = {
            'track_id': tid,
            'box': box,
            'smoothed_box': [float(c) for c in box],
            'disappeared': 0,
            'predicted_class': None,
            'confidence': 0.0,
            'is_low_match': False,
            'scores': [],
            'last_classified_time': 0.0
        }
        if classify_attire and classifier_func:
            self._classify_track(track, img_w, img_h, img_pil, classifier_func, min_conf_thresh)
        self.tracks[tid] = track

    def _update_track(self, tid, new_box, img_w, img_h, img_pil, classifier_func, classify_attire, min_conf_thresh):
        track = self.tracks[tid]
        track['disappeared'] = 0
        track['box'] = new_box
        # Coordinate smoothing (EMA) to prevent box flickering
        alpha = self.ema_alpha
        track['smoothed_box'] = [
            alpha * float(nb) + (1.0 - alpha) * float(sb)
            for nb, sb in zip(new_box, track['smoothed_box'])
        ]

        if classify_attire and classifier_func:
            now = time.time()
            # Classify if never classified or if more than 1.0 second elapsed since last classification
            if track['predicted_class'] is None or (now - track['last_classified_time']) > 1.0:
                self._classify_track(track, img_w, img_h, img_pil, classifier_func, min_conf_thresh)

    def _classify_track(self, track, img_w, img_h, img_pil, classifier_func, min_conf_thresh):
        box = [int(round(c)) for c in track['smoothed_box']]
        exp_box = expand_box_with_padding(box, img_w, img_h, pad_pct=0.04)
        crop_pil = img_pil.crop((exp_box[0], exp_box[1], exp_box[2], exp_box[3]))
        top_cls, top_conf, scores = classify_single_crop(crop_pil, classifier_func)
        track['predicted_class'] = top_cls
        track['confidence'] = top_conf
        track['is_low_match'] = bool(top_conf < min_conf_thresh)
        track['scores'] = scores
        track['last_classified_time'] = time.time()

    def _get_next_available_id(self) -> int:
        used_ids = set(self.tracks.keys())
        cand = 1
        while cand in used_ids:
            cand += 1
        return cand

    def _format_output(self, img_w: int, img_h: int, classify_attire: bool) -> List[Dict[str, Any]]:
        output = []
        # Sort by track_id or x1 coordinate
        sorted_tracks = sorted(self.tracks.values(), key=lambda t: t['smoothed_box'][0])
        for t in sorted_tracks:
            tid = t['track_id']
            box = [int(round(c)) for c in t['smoothed_box']]
            cls = t['predicted_class'] if classify_attire else None
            conf = t['confidence'] if classify_attire else 0.0
            meta = CATEGORY_META.get(cls, DEFAULT_META) if cls else DEFAULT_META

            output.append({
                'person_id': tid,
                'person_label': f"Person {tid}",
                'box': box,
                'box_normalized': [
                    round(box[0] / img_w, 4),
                    round(box[1] / img_h, 4),
                    round(box[2] / img_w, 4),
                    round(box[3] / img_h, 4)
                ],
                'predicted_class': cls,
                'confidence': conf,
                'is_low_match': t['is_low_match'] if classify_attire else False,
                'color': meta['hex'],
                'color_rgb': meta['rgb'],
                'color_bgr': meta['bgr'],
                'emoji': meta.get('emoji', '👤'),
                'desc': meta.get('desc', 'Person detected'),
                'scores': t['scores'] if classify_attire else []
            })
        return output


_live_tracker = PersonTracker()


def get_live_tracker() -> PersonTracker:
    """Return the active singleton PersonTracker instance for live camera."""
    global _live_tracker
    return _live_tracker


def track_and_process_live_frame(
    img: Image.Image,
    classifier_func=None,
    classify_attire: bool = True,
    min_detection_conf: float = 0.35,
    min_clothing_conf_threshold: float = 45.0
) -> Dict[str, Any]:
    """
    Real-time tracking pipeline for live camera streams.
    Detects all individuals, maintains persistent Person IDs across movement,
    smooths bounding boxes to prevent jitter, and conditionally runs clothing
    classification only when classify_attire is True.
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')

    w, h = img.size
    img_np = np.array(img)

    # Detect raw bounding boxes using YOLOv8n
    boxes = detect_person_boxes(img_np, min_conf=min_detection_conf)

    # Update tracker
    persons = _live_tracker.update(
        new_boxes=boxes,
        img_w=w,
        img_h=h,
        img_pil=img,
        classifier_func=classifier_func,
        classify_attire=classify_attire,
        min_clothing_conf_threshold=min_clothing_conf_threshold
    )

    lead_person = persons[0] if persons else {
        'person_id': 1,
        'person_label': 'Person 1',
        'predicted_class': None,
        'confidence': 0.0,
        'emoji': '👤',
        'desc': 'No person currently in frame',
        'color': DEFAULT_META['hex'],
        'scores': []
    }

    return {
        'success': True,
        'total_persons': len(persons),
        'persons': persons,
        'classify_attire': classify_attire,
        'predicted_class': lead_person['predicted_class'],
        'confidence': lead_person['confidence'],
        'emoji': lead_person['emoji'],
        'desc': lead_person['desc'],
        'color': lead_person['color'],
        'scores': lead_person['scores']
    }


def process_multi_person_image(
    img: Image.Image,
    classifier_func,
    min_detection_conf: float = 0.35,
    min_clothing_conf_threshold: float = 45.0,
    classify_attire: bool = True
) -> Dict[str, Any]:
    """
    Complete multi-person pipeline (used for Captured Photo mode and Uploads):
      1. Detect all persons in the image.
      2. If 0 persons detected, fall back to whole image.
      3. Crop each person and classify clothing.
      4. Annotate image with colored bounding boxes and labels.
      5. Return structured result.
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')

    w, h = img.size
    img_np = np.array(img)

    # 1. Detect person bounding boxes
    boxes = detect_person_boxes(img_np, min_conf=min_detection_conf)

    persons = []

    if not boxes:
        # Fallback: No persons detected; classify whole frame
        logger.info("No distinct person bounding boxes detected. Running full-frame inference fallback.")
        top_cls, top_conf, scores = classify_single_crop(img, classifier_func)
        meta = CATEGORY_META.get(top_cls, DEFAULT_META)
        is_low = bool(top_conf < min_clothing_conf_threshold)

        person_data = {
            'person_id': 1,
            'person_label': 'Person 1',
            'box': [0, 0, w, h],
            'box_normalized': [0.0, 0.0, 1.0, 1.0],
            'predicted_class': top_cls if classify_attire else None,
            'confidence': top_conf if classify_attire else 0.0,
            'is_low_match': is_low if classify_attire else False,
            'color': meta['hex'] if classify_attire else DEFAULT_META['hex'],
            'color_rgb': meta['rgb'] if classify_attire else DEFAULT_META['rgb'],
            'color_bgr': meta['bgr'] if classify_attire else DEFAULT_META['bgr'],
            'emoji': meta['emoji'],
            'desc': meta['desc'],
            'scores': scores if classify_attire else [],
            'is_fallback_full_frame': True
        }
        persons.append(person_data)

    else:
        # Process each detected person
        for idx, box in enumerate(boxes):
            expanded_box = expand_box_with_padding(box, w, h, pad_pct=0.04)
            crop_pil = img.crop((expanded_box[0], expanded_box[1], expanded_box[2], expanded_box[3]))

            if classify_attire:
                top_cls, top_conf, scores = classify_single_crop(crop_pil, classifier_func)
                meta = CATEGORY_META.get(top_cls, DEFAULT_META)
                is_low = bool(top_conf < min_clothing_conf_threshold)
            else:
                top_cls = None
                top_conf = 0.0
                scores = []
                meta = DEFAULT_META
                is_low = False

            person_data = {
                'person_id': idx + 1,
                'person_label': f"Person {idx + 1}",
                'box': box,
                'box_normalized': [
                    round(box[0] / w, 4),
                    round(box[1] / h, 4),
                    round(box[2] / w, 4),
                    round(box[3] / h, 4)
                ],
                'predicted_class': top_cls,
                'confidence': top_conf,
                'is_low_match': is_low,
                'color': meta['hex'],
                'color_rgb': meta['rgb'],
                'color_bgr': meta['bgr'],
                'emoji': meta.get('emoji', '👤'),
                'desc': meta.get('desc', 'Person detected'),
                'scores': scores,
                'is_fallback_full_frame': False
            }
            persons.append(person_data)

    # 2. Annotate image with colored bounding boxes
    annotated_pil = annotate_image_with_detections(img, persons, classify_attire=classify_attire)
    annotated_b64 = pil_to_base64_jpeg(annotated_pil)

    # Primary (first) person for legacy / backward compatibility
    lead_person = persons[0]

    return {
        'success': True,
        'total_persons': len(persons),
        'persons': persons,
        'classify_attire': classify_attire,
        'annotated_image': annotated_b64,
        # Legacy compatible fields
        'predicted_class': lead_person['predicted_class'],
        'confidence': lead_person['confidence'],
        'emoji': lead_person['emoji'],
        'desc': lead_person['desc'],
        'color': lead_person['color'],
        'scores': lead_person['scores']
    }
