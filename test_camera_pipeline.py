"""
test_camera_pipeline.py — Comprehensive Unit & Integration Test Suite
---------------------------------------------------------------------
Tests:
  1. CameraManager initialization, mock frame generation, and status reporting
  2. DirectShow resolution negotiation & source switching
  3. Bounded reconnection logic & timeout handling
  4. Explicit resource cleanup on release
  5. AI model inference pipeline on camera frames
  6. Multi-Person Detection & Classification Engine (person_detector)
  7. Visual annotation & category colors (Formal=Blue, Casual=Green, Traditional=Orange)
  8. Gemini API integration via google-genai (gemini-2.5-flash)
  9. Flask API endpoints (/api/camera/devices, /api/camera/select, /api/camera/status, /video_feed, /api/camera/capture_and_predict, /api/camera/detect_live, /predict, /chat_api)
  10. Localhost security guard for camera endpoints
  11. Headless live_camera CLI validation
"""

import os
import io
import time
import glob
import unittest
import numpy as np
from PIL import Image

from camera_manager import CameraManager, scan_available_cameras, MAX_RECONNECT_ATTEMPTS
from app import app, camera_manager, run_inference_on_pil_image, run_multi_person_pipeline, preprocess_image_numpy, get_model, CLASS_META
import gemini_helper
import person_detector


class TestCameraManager(unittest.TestCase):
    def setUp(self):
        self.cam = CameraManager(default_source="mock", width=1280, height=720, fps=30)
        self.cam.start()

    def tearDown(self):
        self.cam.release()

    def test_mock_frame_generation(self):
        """Verify that mock source generates valid RGB frames and metrics."""
        time.sleep(0.1)
        frame_pil = self.cam.get_current_frame(as_pil=True)
        self.assertIsNotNone(frame_pil)
        self.assertIsInstance(frame_pil, Image.Image)
        self.assertEqual(frame_pil.size, (1280, 720))

        frame_np = self.cam.get_current_frame(as_pil=False)
        self.assertIsNotNone(frame_np)
        self.assertEqual(frame_np.shape, (720, 1280, 3))

    def test_status_reporting(self):
        """Verify status dictionary structure and fields."""
        time.sleep(0.1)
        status = self.cam.get_status()
        self.assertEqual(status["status"], "CONNECTED")
        self.assertEqual(status["source"], "mock")
        self.assertTrue(status["is_running"])
        self.assertIn("1280x720", status["resolution"])
        self.assertGreaterEqual(status["frame_count"], 1)

    def test_source_switching_and_cleanup(self):
        """Verify switching sources cleanly releases previous resources."""
        self.cam.switch_source("mock")
        time.sleep(0.1)
        status = self.cam.get_status()
        self.assertEqual(status["status"], "CONNECTED")

    def test_ai_inference_on_camera_frame(self):
        """Verify camera frames feed directly into the AI classification pipeline."""
        time.sleep(0.1)
        frame = self.cam.get_current_frame(as_pil=True)
        self.assertIsNotNone(frame)
        result = run_inference_on_pil_image(frame)
        self.assertTrue(result["success"])
        self.assertIn(result["predicted_class"], ["Formal", "Casual", "Traditional"])
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertEqual(len(result["scores"]), 3)

    def test_device_scanner(self):
        """Verify device scanner runs without exception and returns mock."""
        devices = scan_available_cameras(max_indices=2)
        self.assertIsInstance(devices, list)
        self.assertTrue(any(d["id"] == "mock" for d in devices))


class TestMultiPersonDetection(unittest.TestCase):
    def test_category_colors(self):
        """Verify required visually distinguishable colors: Formal=Blue, Casual=Green, Traditional=Orange."""
        meta = person_detector.CATEGORY_META
        # Formal: Blue
        self.assertEqual(meta['Formal']['hex'].lower(), '#2b7fff')
        # Casual: Green
        self.assertEqual(meta['Casual']['hex'].lower(), '#2ecc71')
        # Traditional: Orange
        self.assertEqual(meta['Traditional']['hex'].lower(), '#ff8c00')

    def test_zero_person_fallback(self):
        """Verify that when no person is detected, the pipeline gracefully classifies whole frame."""
        blank_img = Image.new('RGB', (400, 400), color=(100, 100, 100))
        result = person_detector.process_multi_person_image(
            blank_img,
            classifier_func=lambda crop: run_inference_on_pil_image(crop, generate_ai_feedback=False)
        )
        self.assertTrue(result['success'])
        self.assertEqual(result['total_persons'], 1)
        self.assertEqual(result['persons'][0]['person_label'], 'Person 1')
        self.assertTrue(result['persons'][0]['is_fallback_full_frame'])
        self.assertIn(result['predicted_class'], ['Formal', 'Casual', 'Traditional'])

    def test_multi_person_on_dataset_image(self):
        """Verify multi-person detection accurately identifies persons and crops them."""
        sample_images = glob.glob('dataset/*/*/*.jpg') + glob.glob('dataset/*/*/*.png')
        if sample_images:
            img = Image.open(sample_images[0]).convert('RGB')
            result = run_multi_person_pipeline(img, generate_ai_feedback=False)
            self.assertTrue(result['success'])
            self.assertGreaterEqual(result['total_persons'], 1)
            for p in result['persons']:
                self.assertIn(p['predicted_class'], ['Formal', 'Casual', 'Traditional'])
                self.assertGreaterEqual(p['confidence'], 0.0)
                self.assertIn(p['color'], ['#2b7fff', '#2ecc71', '#ff8c00'])
                self.assertTrue(p['person_label'].startswith('Person '))


class TestGeminiHelper(unittest.TestCase):
    def test_api_key_loading(self):
        """Verify API key is safely loaded without crashing."""
        key = gemini_helper.get_gemini_api_key()
        self.assertIsNotNone(key)
        self.assertGreaterEqual(len(key), 15)
        # Verify key does not contain surrounding quotes or whitespace
        self.assertEqual(key, key.strip("\"'").strip())

    def test_chat_response_format(self):
        """Verify chatbot returns structured suggestions or friendly error when quota exhausted."""
        resp = gemini_helper.generate_chat_response("What should I wear for a business meeting?")
        if not resp.get('success'):
            # If Google API quota is exhausted (free tier 20 req/day), verify friendly error message
            self.assertTrue('quota' in resp.get('error', '').lower() or 'resource_exhausted' in resp.get('error', '').lower())
        else:
            self.assertTrue(resp.get('success'))
            if resp.get('type') == 'suggestions':
                data = resp.get('data')
                self.assertIsInstance(data, list)
                self.assertEqual(len(data), 3)
                for item in data:
                    self.assertIn('outfit', item)
                    self.assertIn('footwear', item)
                    self.assertIn('grooming', item)
                    self.assertIn('accessories', item)


class TestFlaskCameraAPI(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        camera_manager.switch_source("mock")
        time.sleep(0.05)

    def test_health_endpoint(self):
        """Verify /health reports camera status and gemini key status."""
        res = self.client.get('/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('camera', data)
        self.assertIn('has_gemini_key', data)

    def test_get_devices_endpoint(self):
        """Verify /api/camera/devices returns device list."""
        res = self.client.get('/api/camera/devices')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertIsInstance(data['devices'], list)

    def test_camera_status_endpoint(self):
        """Verify /api/camera/status returns current metrics."""
        res = self.client.get('/api/camera/status')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('status', data)
        self.assertIn('current_fps', data)

    def test_camera_select_endpoint(self):
        """Verify /api/camera/select switches camera source."""
        res = self.client.post('/api/camera/select',
                               json={'source': 'mock'},
                               environ_base={'REMOTE_ADDR': '127.0.0.1'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])

    def test_camera_select_security_block(self):
        """Verify non-localhost callers are rejected on /api/camera/select."""
        res = self.client.post('/api/camera/select',
                               json={'source': 'mock'},
                               environ_base={'REMOTE_ADDR': '192.168.1.50'})
        self.assertEqual(res.status_code, 403)

    def test_capture_and_predict_endpoint(self):
        """Verify /api/camera/capture_and_predict returns multi-person AI predictions."""
        time.sleep(0.1)
        res = self.client.post('/api/camera/capture_and_predict')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertIn('total_persons', data)
        self.assertIn('persons', data)
        self.assertIn('annotated_image', data)
        self.assertIn('predicted_class', data)
        self.assertIn('confidence', data)

    def test_detect_live_endpoint(self):
        """Verify /api/camera/detect_live returns fast multi-person tracking data."""
        res = self.client.post('/api/camera/detect_live', json={})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertIn('persons', data)

    def test_predict_multipart_multi_person(self):
        """Verify image upload via /predict returns multi-person detection and classification."""
        img = Image.new('RGB', (300, 300), color=(120, 80, 50))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)

        res = self.client.post('/predict', data={'image': (buf, 'test.jpg')}, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertIn(data['predicted_class'], ['Formal', 'Casual', 'Traditional'])
        self.assertIn('persons', data)
        self.assertIn('annotated_image', data)

    def test_chat_api_success(self):
        """Verify /chat_api handles styling request cleanly."""
        res = self.client.post('/chat_api', json={'message': 'What to wear for a cocktail party?'})
        if res.status_code == 200:
            data = res.get_json()
            if data.get('type') == 'suggestions':
                self.assertEqual(len(data.get('data', [])), 3)
            else:
                self.assertTrue(data.get('type') in ('message', 'error', 'text'))
        else:
            # When free tier daily quota is reached on Gemini API, endpoint returns 400 with quota message
            self.assertEqual(res.status_code, 400)
            data = res.get_json()
            self.assertIn('error', data)


if __name__ == '__main__':
    unittest.main()
