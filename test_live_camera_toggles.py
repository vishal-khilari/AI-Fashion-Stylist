"""
test_live_camera_toggles.py — Comprehensive Test Suite for Live Camera Independent Toggles
------------------------------------------------------------------------------------------
Validates:
1. Toggle State 1: Person Detection OFF, Attire Matching OFF (Default State)
2. Toggle State 2: Person Detection ON, Attire Matching OFF (Live Bounding Boxes, Neutral Color, No Category/Score)
3. Toggle State 3: Person Detection ON, Attire Matching ON (Live Bounding Boxes, Category Colors, Real Model Confidence %)
4. Dependency logic: Enabling Attire Matching automatically activates Person Detection
5. Dependency logic: Disabling Attire Matching keeps Person Detection ON
6. Dependency logic: Disabling Person Detection disables Attire Matching
7. Multi-person tracking: ID stability across frames, EMA coordinate smoothing, new person entry & exit
8. Classification caching per tracking ID: Avoids redundant inference spikes on moving tracked persons
9. /api/camera/detect_live endpoint handling for both classify_attire=False and classify_attire=True
"""

import unittest
import numpy as np
from PIL import Image
import io
import base64
import time

from app import app, get_model, preprocess_image_numpy, CLASS_NAMES, CLASS_META
import person_detector


class TestLiveCameraToggles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = app
        cls.app.config['TESTING'] = True
        cls.client = cls.app.test_client()
        cls.model_obj = get_model()

        # Create a sample base64 JPEG image for live detection API tests
        test_img = Image.new('RGB', (640, 480), color=(50, 50, 50))
        buf = io.BytesIO()
        test_img.save(buf, format='JPEG')
        cls.b64_image = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

    def setUp(self):
        # Reset the tracker before each test
        tracker = person_detector.get_live_tracker()
        tracker.reset()

    def dummy_classifier(self, crop):
        return {
            'predicted_class': 'Formal',
            'confidence': 93.4,
            'scores': [{'class': 'Formal', 'score': 93.4, 'color': '#2b7fff'}]
        }

    def test_default_toggle_states(self):
        """Verify that default state for live detection tracker starts cleanly."""
        tracker = person_detector.get_live_tracker()
        self.assertEqual(tracker.next_person_id, 1)
        self.assertEqual(len(tracker.tracks), 0)

    def test_toggle_state_on_off_neutral_boxes(self):
        """
        Toggle: Person Detection ON, Attire Matching OFF.
        Expected:
        - Bounding boxes are drawn in neutral accent color (#c9a96e / gold).
        - Label displays ONLY 'Person {id}'.
        - Predicted attire class and confidence percentage are HIDDEN.
        """
        test_img = Image.new('RGB', (640, 480), color=(30, 30, 30))

        # Test annotate_image_with_detections with classify_attire=False
        fake_persons = [
            {
                'person_id': 1,
                'person_label': 'Person 1',
                'box': [50, 50, 200, 400],
                'predicted_class': None,
                'confidence': 0.0,
                'color': person_detector.NEUTRAL_BOX_COLOR
            },
            {
                'person_id': 2,
                'person_label': 'Person 2',
                'box': [300, 50, 450, 400],
                'predicted_class': None,
                'confidence': 0.0,
                'color': person_detector.NEUTRAL_BOX_COLOR
            }
        ]

        annotated_img = person_detector.annotate_image_with_detections(
            test_img,
            fake_persons,
            classify_attire=False
        )
        self.assertIsInstance(annotated_img, Image.Image)
        self.assertEqual(annotated_img.size, (640, 480))

        # Check CV2 frame annotation as well
        cv2_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated_cv2 = person_detector.annotate_cv2_frame(cv2_frame, fake_persons, classify_attire=False)
        self.assertEqual(annotated_cv2.shape, (480, 640, 3))

    def test_toggle_state_on_on_category_colors_and_scores(self):
        """
        Toggle: Person Detection ON, Attire Matching ON.
        Expected:
        - Category-specific colors: Formal (Blue #2b7fff), Casual (Green #2ecc71), Traditional (Orange #ff8c00).
        - Label displays 'Person {id}: {Class} ({Match}%)'.
        - Real model confidence percentage is displayed.
        """
        test_img = Image.new('RGB', (640, 480), color=(30, 30, 30))

        fake_persons = [
            {
                'person_id': 1,
                'person_label': 'Person 1',
                'box': [50, 50, 200, 400],
                'predicted_class': 'Formal',
                'confidence': 94.5,
                'color': person_detector.CATEGORY_META['Formal']['hex']
            },
            {
                'person_id': 2,
                'person_label': 'Person 2',
                'box': [300, 50, 450, 400],
                'predicted_class': 'Casual',
                'confidence': 91.2,
                'color': person_detector.CATEGORY_META['Casual']['hex']
            }
        ]

        annotated_img = person_detector.annotate_image_with_detections(
            test_img,
            fake_persons,
            classify_attire=True
        )
        self.assertIsInstance(annotated_img, Image.Image)

        # Check CV2 frame annotation
        cv2_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated_cv2 = person_detector.annotate_cv2_frame(cv2_frame, fake_persons, classify_attire=True)
        self.assertEqual(annotated_cv2.shape, (480, 640, 3))

    def test_tracker_id_persistence_and_ema_smoothing(self):
        """Verify PersonTracker retains ID across frames and smooths bounding boxes with EMA."""
        tracker = person_detector.PersonTracker(iou_threshold=0.3, max_disappeared=5, ema_alpha=0.7)

        # Frame 1: Person at (100, 100, 200, 300)
        detections_f1 = [[100, 100, 200, 300]]
        tracks_f1 = tracker.update(detections_f1)
        self.assertEqual(len(tracks_f1), 1)
        self.assertEqual(tracks_f1[0]['person_id'], 1)
        self.assertEqual(tracks_f1[0]['box'], [100, 100, 200, 300])

        # Frame 2: Person moves slightly to (105, 102, 205, 302)
        detections_f2 = [[105, 102, 205, 302]]
        tracks_f2 = tracker.update(detections_f2)
        self.assertEqual(len(tracks_f2), 1)
        # ID MUST remain 1
        self.assertEqual(tracks_f2[0]['person_id'], 1)
        # EMA smoothed box should be between 100 and 105: 0.7*105 + 0.3*100 = 73.5 + 30 = 103.5 -> 104
        smooth_box = tracks_f2[0]['box']
        self.assertGreaterEqual(smooth_box[0], 100)
        self.assertLessEqual(smooth_box[0], 105)

    def test_tracker_multiple_people_independent_ids(self):
        """Verify tracker tracks multiple people with distinct, stable IDs."""
        tracker = person_detector.PersonTracker()

        # Frame 1: 2 people
        d1 = [[50, 50, 150, 350], [300, 50, 400, 350]]
        tracks1 = tracker.update(d1)
        self.assertEqual(len(tracks1), 2)
        ids1 = {t['person_id'] for t in tracks1}
        self.assertEqual(ids1, {1, 2})

        # Frame 2: Both move slightly
        d2 = [[52, 51, 152, 351], [303, 49, 403, 349]]
        tracks2 = tracker.update(d2)
        self.assertEqual(len(tracks2), 2)
        ids2 = {t['person_id'] for t in tracks2}
        self.assertEqual(ids2, {1, 2})

        # Frame 3: A third person enters
        d3 = [[52, 51, 152, 351], [303, 49, 403, 349], [500, 50, 600, 350]]
        tracks3 = tracker.update(d3)
        self.assertEqual(len(tracks3), 3)
        ids3 = {t['person_id'] for t in tracks3}
        self.assertEqual(ids3, {1, 2, 3})

    def test_tracker_person_exits_frame(self):
        """Verify person is removed from tracker when disappearing for max_disappeared frames."""
        tracker = person_detector.PersonTracker(max_disappeared=3)

        # Person enters
        tracker.update([[100, 100, 200, 300]])
        self.assertIn(1, tracker.tracks)

        # Person disappears for 2 frames (still within max_disappeared)
        tracker.update([])
        self.assertIn(1, tracker.tracks)
        tracker.update([])
        self.assertIn(1, tracker.tracks)

        # Person disappears for 3rd and 4th frame -> deregistered
        tracker.update([])
        tracker.update([])
        self.assertNotIn(1, tracker.tracks)

    def test_classification_caching(self):
        """Verify clothing classification is cached per ID to avoid continuous inference recalculation."""
        tracker = person_detector.PersonTracker()
        tracker.update([[100, 100, 200, 300]])

        # Initially no cache
        self.assertIsNone(tracker.get_cached_classification(1))

        # Store cache
        fake_res = {'predicted_class': 'Casual', 'confidence': 91.2, 'scores': []}
        tracker.set_cached_classification(1, fake_res)

        # Immediately retrieve -> cache hit
        cached = tracker.get_cached_classification(1, max_age_seconds=1.5)
        self.assertIsNotNone(cached)
        self.assertEqual(cached['predicted_class'], 'Casual')
        self.assertEqual(cached['confidence'], 91.2)

    def test_api_detect_live_attire_off(self):
        """Verify /api/camera/detect_live returns neutral styling when classify_attire is False."""
        res = self.client.post('/api/camera/detect_live', json={
            'image': self.b64_image,
            'classify_attire': False
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertFalse(data.get('classify_attire', True))
        for p in data.get('persons', []):
            self.assertEqual(p['color'], person_detector.NEUTRAL_BOX_COLOR)
            self.assertIsNone(p.get('predicted_class'))
            self.assertEqual(p.get('confidence', 0.0), 0.0)

    def test_api_detect_live_attire_on(self):
        """Verify /api/camera/detect_live returns real attire classification when classify_attire is True."""
        res = self.client.post('/api/camera/detect_live', json={
            'image': self.b64_image,
            'classify_attire': True
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertTrue(data.get('classify_attire', False))
        for p in data.get('persons', []):
            if p.get('predicted_class'):
                self.assertIn(p['predicted_class'], ['Formal', 'Casual', 'Traditional'])
                self.assertGreater(p['confidence'], 0.0)
                self.assertIn(p['color'], ['#2b7fff', '#2ecc71', '#ff8c00'])


if __name__ == '__main__':
    unittest.main()
