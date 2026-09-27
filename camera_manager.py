"""
camera_manager.py — Thread-Safe Real-Time USB / Mobile Camera Pipeline
----------------------------------------------------------------------
Manages live video capture from:
  1. USB-connected mobile cameras via DirectShow index (DroidCam, Iriun, Camo, Android UVC)
  2. Local USB / ADB IP streams (e.g., http://127.0.0.1:8080/video, http://127.0.0.1:4747/video)
  3. Built-in / External Webcams (Index 0, 1, 2...)
  4. Synthetic "mock" test camera for offline / CI verification

Key Features:
  - Non-blocking background worker thread with low-latency frame buffer
  - DirectShow resolution and FPS negotiation (1280x720 @ 30fps default)
  - Bounded reconnection with explicit timeout (max 15 retries / 30 seconds)
  - Strict resource cleanup on source switch & process shutdown (atexit)
  - Device auto-discovery for hardware indices and local ADB ports
"""

import cv2
import time
import socket
import atexit
import logging
import threading
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Reconnection & capture constants
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
MAX_RECONNECT_ATTEMPTS = 15
RECONNECT_INTERVAL_SEC = 2.0


def is_port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    """Check if a local TCP port is open (used to detect DroidCam/IP Webcam ADB ports)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def scan_available_cameras(max_indices: int = 4):
    """
    Scan the system for available video capture devices and known local USB/ADB stream endpoints.
    Returns a list of device descriptors.
    """
    devices = []

    # 1. Scan DirectShow camera indices
    for idx in range(max_indices):
        cap = None
        try:
            # Try DirectShow first on Windows
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(idx)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    # DirectShow index 0 is typically built-in, 1+ are usually USB/external/virtual
                    label = f"Camera {idx}"
                    desc = f"DirectShow Video Device {idx} ({w}x{h})"
                    if idx >= 1:
                        desc += " [Likely USB / Mobile Camera]"
                    devices.append({
                        "id": str(idx),
                        "type": "index",
                        "index": idx,
                        "label": label,
                        "description": desc,
                        "resolution": f"{w}x{h}"
                    })
        except Exception as e:
            logger.debug(f"Device index {idx} check error: {e}")
        finally:
            if cap is not None:
                cap.release()

    # 2. Check for common local ADB/USB streaming ports
    # DroidCam default port: 4747
    if is_port_open("127.0.0.1", 4747):
        devices.append({
            "id": "http://127.0.0.1:4747/video",
            "type": "stream",
            "url": "http://127.0.0.1:4747/video",
            "label": "DroidCam USB (127.0.0.1:4747)",
            "description": "DroidCam live stream via USB/ADB port 4747"
        })

    # IP Webcam default port: 8080
    if is_port_open("127.0.0.1", 8080):
        devices.append({
            "id": "http://127.0.0.1:8080/video",
            "type": "stream",
            "url": "http://127.0.0.1:8080/video",
            "label": "IP Webcam USB (127.0.0.1:8080)",
            "description": "IP Webcam MJPEG stream via USB/ADB port 8080"
        })

    # 3. Always provide synthetic mock source for testing & verification
    devices.append({
        "id": "mock",
        "type": "mock",
        "label": "Mock Camera (Synthetic Test Source)",
        "description": "Generates synthetic test frames for offline/CI testing without physical camera"
    })

    return devices


class CameraManager:
    """
    Thread-safe camera capture manager with bounded reconnection,
    explicit resolution negotiation, and clean resource lifecycle.
    """

    def __init__(self, default_source=None, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, fps=DEFAULT_FPS):
        self.width = width
        self.height = height
        self.target_fps = fps
        self.lock = threading.Lock()
        
        self.source = default_source
        self.cap = None
        self.is_running = False
        self.thread = None
        
        self.latest_frame = None
        self.latest_frame_time = 0.0
        self.frame_count = 0
        self.current_fps = 0.0
        
        self.status = "IDLE"  # "IDLE", "CONNECTING", "CONNECTED", "RECONNECTING", "DISCONNECTED", "ERROR"
        self.error_message = None
        self.reconnect_attempts = 0

        # Auto-register cleanup on process exit
        atexit.register(self.release)

    def _create_mock_frame(self, count: int) -> np.ndarray:
        """Generate a clean synthetic test frame with simulated clothing shapes and timestamp."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Elegant dark backdrop gradient
        for y in range(self.height):
            c = int(18 + (y / self.height) * 25)
            frame[y, :] = [c, c, c]

        # Draw grid pattern
        cv2.rectangle(frame, (40, 40), (self.width - 40, self.height - 40), (45, 45, 45), 1)

        # Draw a stylish clothing silhouette / dummy fashion mannequin
        cx, cy = self.width // 2, self.height // 2
        # Torso
        cv2.rectangle(frame, (cx - 100, cy - 120), (cx + 100, cy + 160), (90, 80, 70), -1)
        # Collar / Lapel
        pts_left = np.array([[cx - 70, cy - 120], [cx, cy - 20], [cx - 20, cy - 120]], np.int32)
        pts_right = np.array([[cx + 70, cy - 120], [cx, cy - 20], [cx + 20, cy - 120]], np.int32)
        cv2.fillPoly(frame, [pts_left, pts_right], (201, 169, 110))
        # Tie
        tie_pts = np.array([[cx - 15, cy - 20], [cx + 15, cy - 20], [cx + 22, cy + 80], [cx, cy + 120], [cx - 22, cy + 80]], np.int32)
        cv2.fillPoly(frame, [tie_pts], (60, 60, 180))

        # Dynamic sine-wave indicator
        sine_offset = int(np.sin(count * 0.1) * 30)
        cv2.circle(frame, (cx + sine_offset, cy - 170), 35, (180, 160, 140), -1)

        # Header metadata
        cv2.putText(frame, "MOCK USB MOBILE CAMERA STREAM", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (201, 169, 110), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {count} | Source: {self.source} | Target: 1280x720", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

        return frame

    def _open_capture(self, src):
        """Open a video capture device or stream with explicit resolution/FPS negotiation."""
        if src == "mock":
            return True

        # Parse source: integer device index or string URL
        idx = None
        if isinstance(src, int):
            idx = src
        elif isinstance(src, str) and src.isdigit():
            idx = int(src)

        cap = None
        try:
            if idx is not None:
                # Use DirectShow backend on Windows for reliable USB device capture
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(idx)
            else:
                # URL stream (MJPEG / RTSP / HTTP)
                cap = cv2.VideoCapture(src)

            if not cap.isOpened():
                if cap is not None:
                    cap.release()
                return False

            # Explicitly negotiate resolution and FPS properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
            # Prefer MJPG fourcc for USB webcams to allow higher resolutions at 30fps
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception:
                pass

            # Read a test frame to ensure stream is actually active
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                return False

            self.cap = cap
            return True

        except Exception as e:
            logger.error(f"Error opening capture source '{src}': {e}")
            if cap is not None:
                cap.release()
            return False

    def _worker_loop(self):
        """Dedicated background thread continuously polling frames."""
        last_calc_time = time.time()
        fps_frame_count = 0
        frame_interval = 1.0 / max(1, self.target_fps)

        while self.is_running:
            start_time = time.time()

            if self.source == "mock":
                # Generate synthetic test frame
                with self.lock:
                    self.status = "CONNECTED"
                    self.error_message = None
                    self.frame_count += 1
                    fps_frame_count += 1
                    self.latest_frame = self._create_mock_frame(self.frame_count)
                    self.latest_frame_time = time.time()
            else:
                if self.cap is None or not self.cap.isOpened():
                    # Attempt bounded reconnection
                    reconnected = self._handle_reconnection()
                    if not reconnected:
                        # Reconnection timed out or aborted
                        break
                    continue

                try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        with self.lock:
                            self.status = "CONNECTED"
                            self.error_message = None
                            self.reconnect_attempts = 0
                            self.latest_frame = frame
                            self.latest_frame_time = time.time()
                            self.frame_count += 1
                            fps_frame_count += 1
                    else:
                        # Read failure -> trigger reconnection
                        logger.warning("Failed to read frame from capture device. Triggering reconnection...")
                        with self.lock:
                            self.status = "RECONNECTING"
                            if self.cap is not None:
                                self.cap.release()
                                self.cap = None
                except Exception as e:
                    logger.error(f"Exception reading frame from camera: {e}")
                    with self.lock:
                        self.status = "RECONNECTING"
                        if self.cap is not None:
                            self.cap.release()
                            self.cap = None

            # Calculate actual FPS periodically
            now = time.time()
            if now - last_calc_time >= 1.0:
                self.current_fps = fps_frame_count / (now - last_calc_time)
                fps_frame_count = 0
                last_calc_time = now

            # Throttle loop to target FPS rate to save CPU
            elapsed = time.time() - start_time
            sleep_time = max(0.001, frame_interval - elapsed)
            time.sleep(sleep_time)

        # Cleanup if worker thread exits
        self._close_capture_handle()

    def _handle_reconnection(self) -> bool:
        """
        Bounded reconnection logic: retries every RECONNECT_INTERVAL_SEC
        up to MAX_RECONNECT_ATTEMPTS (30 seconds total).
        """
        with self.lock:
            self.status = "RECONNECTING"
            self.reconnect_attempts += 1
            current_attempt = self.reconnect_attempts

        if current_attempt > MAX_RECONNECT_ATTEMPTS:
            with self.lock:
                self.status = "DISCONNECTED"
                self.error_message = (
                    f"Camera disconnected: Reconnection timed out after {MAX_RECONNECT_ATTEMPTS} attempts. "
                    "Please check USB connection or select another camera."
                )
            logger.error(self.error_message)
            return False

        logger.info(f"Reconnection attempt {current_attempt}/{MAX_RECONNECT_ATTEMPTS} for source: {self.source}")
        time.sleep(RECONNECT_INTERVAL_SEC)

        if not self.is_running:
            return False

        success = self._open_capture(self.source)
        if success:
            with self.lock:
                self.status = "CONNECTED"
                self.error_message = None
                self.reconnect_attempts = 0
            logger.info(f"Successfully reconnected to camera source: {self.source}")
            return True
        return False

    def _close_capture_handle(self):
        """Safely release the cv2.VideoCapture handle without holding locks long."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"Error releasing VideoCapture: {e}")
            finally:
                self.cap = None

    def start(self, source=None) -> bool:
        """Start capturing from the specified or current source."""
        self.release()

        if source is not None:
            self.source = source
        if self.source is None:
            # Default to scanning and picking the best available camera
            devices = scan_available_cameras()
            # Prioritize USB / External camera (index >= 1 or stream), else index 0, else mock
            usb_devs = [d for d in devices if d.get("index", 0) >= 1 or d["type"] == "stream"]
            if usb_devs:
                self.source = usb_devs[0]["id"]
            elif any(d["type"] == "index" for d in devices):
                self.source = "0"
            else:
                self.source = "mock"

        logger.info(f"Starting CameraManager with source: {self.source}")
        self.status = "CONNECTING"
        self.error_message = None
        self.reconnect_attempts = 0

        # Initial open attempt
        opened = self._open_capture(self.source)
        if not opened and self.source != "mock":
            self.status = "DISCONNECTED"
            self.error_message = f"Could not connect to camera source '{self.source}'."
            logger.warning(self.error_message)
            # We still allow the worker to run if user wants bounded reconnect, but if immediate fail, start thread
        else:
            self.status = "CONNECTED"

        self.is_running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True, name="CameraWorker")
        self.thread.start()
        return True

    def switch_source(self, new_source) -> bool:
        """Switch to a new camera source, cleanly releasing old capture handles."""
        logger.info(f"Switching camera source from '{self.source}' to '{new_source}'")
        return self.start(source=new_source)

    def release(self):
        """Stop background worker and release all hardware/stream resources."""
        self.is_running = False
        if self.thread is not None and self.thread.is_alive():
            # Wait briefly for thread to exit
            self.thread.join(timeout=1.0)
            self.thread = None

        self._close_capture_handle()
        with self.lock:
            self.status = "IDLE"
            self.latest_frame = None

    def get_status(self) -> dict:
        """Get the current camera state, active source, resolution, and FPS."""
        with self.lock:
            h, w = (0, 0)
            if self.latest_frame is not None:
                h, w = self.latest_frame.shape[:2]
            return {
                "status": self.status,
                "source": str(self.source),
                "is_running": self.is_running,
                "current_fps": round(self.current_fps, 1),
                "frame_count": self.frame_count,
                "resolution": f"{w}x{h}" if w > 0 else f"{self.width}x{self.height}",
                "reconnect_attempts": self.reconnect_attempts,
                "max_reconnect_attempts": MAX_RECONNECT_ATTEMPTS,
                "error_message": self.error_message
            }

    def get_current_frame(self, as_pil: bool = True):
        """
        Get the freshest frame captured from the camera.
        Returns PIL.Image (RGB) if as_pil=True, else numpy BGR array.
        """
        with self.lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()

        if as_pil:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        return frame

    def generate_mjpeg_stream(self):
        """
        Generator producing multipart/x-mixed-replace JPEG frames
        for streaming via Flask /video_feed route.
        """
        target_delay = 1.0 / max(1, self.target_fps)
        last_sent_time = 0.0

        while self.is_running:
            now = time.time()
            if now - last_sent_time < target_delay:
                time.sleep(0.005)
                continue

            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()

            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    last_sent_time = now
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                # Display a placeholder frame if connecting / disconnected
                placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                msg = f"Camera Status: {self.status}"
                cv2.putText(placeholder, msg, (50, self.height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (201, 169, 110), 2, cv2.LINE_AA)
                ret, jpeg = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    last_sent_time = now
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            time.sleep(0.01)
