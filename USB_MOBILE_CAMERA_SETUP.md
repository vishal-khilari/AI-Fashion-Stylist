# Mobile Phone Camera (DroidCam Wi-Fi & USB) Setup Guide

This guide provides instructions for connecting any **Android** or **iPhone / iOS** device to your laptop using **DroidCam** over **Wi-Fi** or **USB** to use the phone's camera as the real-time input for the **StyleSense AI Fashion Stylist & Dress Classifier**.

---

## Architecture

```text
Android / iOS Phone Camera
       │
       ▼
   DroidCam (over Wi-Fi or USB)
       │
       ▼
DroidCam Windows Client
       │
       ▼
Windows Virtual Camera ("DroidCam Source 2" / "DroidCam Source 3")
       │
       ▼
Browser WebRTC / getUserMedia()
       │
       ▼
StyleSense Website (Live Video & AI Classifier)
```

---

## Connecting with DroidCam (Wi-Fi or USB)

DroidCam turns your phone into a high-definition webcam on Windows. Once the DroidCam Windows client is running, Windows registers a virtual webcam device that any browser can access directly through standard `navigator.mediaDevices.getUserMedia()`.

### Step 1: Install DroidCam
1. **On your PC**: Download and install the **DroidCam Client** for Windows from [dev47apps.com](https://www.dev47apps.com/).
2. **On your Phone**: Install **DroidCam** from Google Play Store (Android) or Apple App Store (iOS).

### Step 2: Connect over Wi-Fi (No Cables Required!)
1. Ensure your phone and laptop are connected to the same Wi-Fi network (or laptop mobile hotspot).
2. Open the DroidCam app on your phone. It will display a **Wi-Fi IP** (e.g., `192.168.1.15`) and **DroidCam Port** (e.g., `4747`).
3. Open the **DroidCam Client** on Windows.
4. Select the **Wi-Fi icon**.
5. Enter the **Device IP** and **Port** shown on your phone.
6. Click **Start**.
7. The phone's camera feed will now appear in the DroidCam desktop client!

### Step 3: Connect over USB (Alternative)
- **Android**:
  1. Enable **Developer Options** > **USB Debugging** on your phone.
  2. Connect the phone to the laptop via USB cable.
  3. In DroidCam Windows Client, click the **USB icon** and click **Start**.
- **iOS**:
  1. Connect your iPhone to your laptop via USB cable and tap **Trust this computer**.
  2. In DroidCam Windows Client, click the **USB icon** and click **Start**.

---

### Method 2: Iriun 4K Webcam USB (Alternative for Android & iOS)

Iriun Webcam offers high resolution (up to 4K) over USB with automatic plug-and-play detection.

1. **On your PC**: Download and install the Windows driver from [iriun.com](https://iriun.com/).
2. **On your Phone**: Install **Iriun Webcam** from Play Store or App Store.
3. Connect your phone to your PC via **USB cable**.
4. Open the Iriun app on both phone and PC. The connection establishes automatically over USB.
5. In StyleSense, select `Camera 1` (or `Iriun Webcam`) from the device dropdown.

---

### Method 3: IP Webcam + ADB USB Port Forwarding (Android)

If you prefer lightweight streaming over USB without installing desktop client software:

1. Install **IP Webcam** from Google Play Store on your Android phone.
2. Enable **USB Debugging** on your phone.
3. In terminal / Command Prompt on your PC, forward the port via ADB:
   ```bash
   adb forward tcp:8080 tcp:8080
   ```
4. In the IP Webcam app on the phone, scroll to the bottom and tap **Start Server**.
5. In StyleSense:
   - Click **Custom Stream URL** in the Live Camera modal.
   - Enter: `http://127.0.0.1:8080/video` and click **Connect**.

---

### Method 4: Android 14+ Native USB Webcam (No Apps Required)

On modern Android devices running Android 14 or higher:

1. Connect the phone to the laptop via **USB cable**.
2. Swipe down the notification shade and tap **Charging this device via USB** (USB Preferences).
3. Select **Webcam**.
4. Windows will recognize the phone as a standard UVC USB Camera device (`Camera 1` or `Camera 2`).
5. Open StyleSense, and the phone camera will appear in the device selector.

---

## How to Start and Verify the System

### 1. Start the Flask Web App
```bash
python app.py
```
Open your browser at `http://localhost:5000`.

1. Click **📱 Real-Time Mobile Camera**.
2. Select your mobile camera device from the dropdown (e.g., `Camera 1 - DirectShow Video Device 1 [Likely USB / Mobile Camera]`).
3. You will immediately see the continuous, low-latency live video stream from your phone.
4. Click **📸 Capture & Classify** to run instant AI dress classification and receive Gemini GenAI styling advice!

---

### 2. Run the Real-Time Desktop Visualizer (CLI)
You can also launch real-time video inference in a dedicated desktop window:
```bash
# Auto-detect camera:
python live_camera.py

# Or specify your mobile camera index directly:
python live_camera.py --source 1

# Or with IP stream:
python live_camera.py --source http://127.0.0.1:8080/video

# Or test with synthetic mock frames (offline mode):
python live_camera.py --source mock
```

**Live Keybindings:**
- `q` / `ESC`: Quit
- `s`: Query Google Gemini AI Stylist in real-time
- `c`: Cycle through available camera devices
- `SPACE`: Freeze / Unfreeze display

---

## Troubleshooting & FAQs

- **Q: What happens if the USB cable is unplugged?**
  - StyleSense includes automatic bounded reconnection (retrying for up to 30 seconds). If connection cannot be restored, the web interface automatically falls back to your laptop/browser webcam so you are never stuck on a dead screen.
- **Q: Why does the camera index show as Camera 0?**
  - Camera 0 is usually the built-in laptop webcam. External USB devices are typically assigned indices `1`, `2`, or higher.
- **Q: How can I test without a physical phone?**
  - Select **Mock Camera (Synthetic Test Source)** from the device dropdown, or run `python live_camera.py --source mock`.
