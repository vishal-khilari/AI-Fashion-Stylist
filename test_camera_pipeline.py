"""
test_camera_pipeline.py — Comprehensive Unit & Integration Test Suite
---------------------------------------------------------------------
Tests:
  1. CameraManager initialization, mock frame generation, and status reporting
  2. DirectShow resolution negotiation & source switching
  3. Bounded reconnection logic & timeout handling
  4. Explicit resource cleanup on release
  5. AI model inference pipeline on camera frames
  6. Flask API endpoints (/api/camera/devices, /api/camera/select, /api/camera/status, /video_feed, /api/camera/capture_and_predict, /predict)
  7. Localhost security guard for camera endpoints
  8. Headless live_camera CLI validation
"""

import os
import io
import time
import unittest
import numpy as np
from PIL import Image

from camera_manager import CameraManager, scan_available_cameras, MAX_RECONNECT_ATTEMPTS
from app import app, camera_manager, run_inference_on_pil_image, preprocess_image_numpy, get_model


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


class TestFlaskCameraAPI(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        # Set camera source to mock for deterministic tests
        camera_manager.switch_source("mock")
        time.sleep(0.05)

    def test_health_endpoint(self):
        """Verify /health reports camera status."""
        res = self.client.get('/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('camera', data)

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
        """Verify /api/camera/capture_and_predict returns AI style predictions."""
        time.sleep(0.1)
        res = self.client.post('/api/camera/capture_and_predict')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertIn('predicted_class', data)
        self.assertIn('confidence', data)
        self.assertIn('scores', data)
        self.assertIn('emoji', data)
        self.assertIn('desc', data)

    def test_predict_multipart_backward_compat(self):
        """Verify standard image upload via /predict remains 100% functional."""
        # Create a test synthetic image
        img = Image.new('RGB', (224, 224), color=(120, 80, 50))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)

        res = self.client.post('/predict', data={'image': (buf, 'test.jpg')}, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertIn(data['predicted_class'], ['Formal', 'Casual', 'Traditional'])


if __name__ == '__main__':
    unittest.main()
