import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import sys
import base64
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten, Dense
from flask import Flask, render_template_string, jsonify, request

app = Flask(__name__)

model = None
face_cascade = None

EMOTIONS = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear']

def build_fer_model(input_shape=(48, 48, 1), classes=7):
    num_features = 64
    m = Sequential()
    m.add(Input(shape=input_shape))
    m.add(Conv2D(num_features, kernel_size=(3, 3)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(Conv2D(num_features, kernel_size=(3, 3)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(Dropout(0.5))

    m.add(Conv2D(num_features, (3, 3), activation='relu'))
    m.add(Conv2D(num_features, (3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    m.add(Conv2D(2 * num_features, kernel_size=(3, 3)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(Conv2D(2 * num_features, kernel_size=(3, 3)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))

    m.add(Conv2D(2 * num_features, (3, 3), activation='relu'))
    m.add(Conv2D(2 * num_features, (3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    m.add(Conv2D(4 * num_features, kernel_size=(3, 3)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(Conv2D(4 * num_features, kernel_size=(3, 3)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))

    m.add(Flatten())
    m.add(Dense(1024, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(1024, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(classes, activation='softmax'))
    return m

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

face_cascade = None
face_cascade_alt = None

def init_resources():
    global model, face_cascade, face_cascade_alt
    weights_path = os.path.join(BASE_DIR, 'top_models', 'fer.h5')
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...", flush=True)
        model = build_fer_model()
        model.load_weights(weights_path)
        print("Model initialized successfully!", flush=True)
    else:
        print(f"Error: Weights file not found at '{weights_path}'!", flush=True)

    cascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
    if os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)
        print("Primary face cascade loaded successfully!", flush=True)

    # Load secondary cascade for improved multi-angle/tilted face detection
    alt_path = getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_alt2.xml'
    if os.path.exists(alt_path):
        face_cascade_alt = cv2.CascadeClassifier(alt_path)
        print("Secondary face cascade loaded successfully!", flush=True)

# Initialize resources automatically on module import (required for Gunicorn/Render)
init_resources()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-Time Facial Emotion Recognition</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0c10;
            --bg-card: rgba(22, 27, 34, 0.75);
            --border-color: rgba(255, 255, 255, 0.1);
            --accent-cyan: #00e5ff;
            --accent-purple: #ab47bc;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(0, 229, 255, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(171, 71, 188, 0.08) 0%, transparent 40%);
        }

        header {
            padding: 1.25rem 2rem;
            background: rgba(10, 12, 16, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo-group {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.2rem;
            color: #000;
            box-shadow: 0 0 15px rgba(0, 229, 255, 0.3);
        }

        h1 {
            font-size: 1.35rem;
            font-weight: 600;
            letter-spacing: -0.02em;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(0, 230, 118, 0.1);
            border: 1px solid rgba(0, 230, 118, 0.3);
            color: #00e676;
            padding: 0.35rem 0.85rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: #00e676;
            border-radius: 50%;
            animation: pulse 1.8s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(0, 230, 118, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0); }
        }

        main {
            flex: 1;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 1.75rem;
        }

        @media (max-width: 1024px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .video-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.25rem;
            backdrop-filter: blur(16px);
            display: flex;
            flex-direction: column;
            gap: 1rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }

        .video-container {
            width: 100%;
            aspect-ratio: 16 / 9;
            background: #000;
            border-radius: 14px;
            overflow: hidden;
            position: relative;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.8);
            border: 1px solid rgba(255, 255, 255, 0.05);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        #videoElement {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transform: scaleX(-1); /* Mirror view */
        }

        #overlayCanvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }

        .camera-prompt {
            position: absolute;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
            color: var(--text-secondary);
        }

        .btn-start {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            color: #000;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 1rem;
            padding: 0.75rem 1.75rem;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            box-shadow: 0 0 20px rgba(0, 229, 255, 0.4);
            transition: all 0.2s ease;
        }

        .btn-start:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 30px rgba(0, 229, 255, 0.6);
        }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }

        .analytics-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.5rem;
            backdrop-filter: blur(16px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }

        .card-header {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .metric-hero {
            background: linear-gradient(135deg, rgba(0, 229, 255, 0.05), rgba(171, 71, 188, 0.05));
            border: 1px solid rgba(0, 229, 255, 0.2);
            border-radius: 14px;
            padding: 1.25rem;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .metric-label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), #ffffff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .metric-confidence {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            color: var(--accent-cyan);
            margin-top: 0.2rem;
        }

        .bars-list {
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
        }

        .bar-item {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .bar-info {
            display: flex;
            justify-content: space-between;
            font-size: 0.88rem;
        }

        .bar-name {
            font-weight: 500;
        }

        .bar-percentage {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
        }

        .bar-track {
            height: 8px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 4px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: 4px;
            width: 0%;
            transition: width 0.3s ease, background-color 0.3s ease;
        }

        footer {
            text-align: center;
            padding: 1rem;
            color: var(--text-secondary);
            font-size: 0.85rem;
            border-top: 1px solid var(--border-color);
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-group">
            <div class="logo-icon">AI</div>
            <h1>Facial Emotion Recognition</h1>
        </div>
        <div class="status-badge" id="statusBadge">
            <div class="pulse-dot"></div>
            <span id="statusText">WEBCAM READY</span>
        </div>
    </header>

    <main>
        <section class="video-card">
            <div class="video-container">
                <video id="videoElement" autoplay playsinline></video>
                <canvas id="overlayCanvas"></canvas>
                <div class="camera-prompt" id="cameraPrompt">
                    <p>Click below to enable your camera and start emotion detection</p>
                    <button class="btn-start" onclick="startCamera()">📷 Enable Camera</button>
                </div>
                <div class="camera-prompt" id="stopControls" style="display:none; top: 15px; right: 15px; left: auto;">
                    <button class="btn-start" style="background: linear-gradient(135deg, #ff1744, #ff5252); color: white;" onclick="stopCamera()">🛑 Stop Camera</button>
                </div>
            </div>
        </section>

        <aside class="sidebar">
            <div class="analytics-card">
                <div class="card-header">
                    <span>Emotion Analytics</span>
                    <span style="font-size:0.8rem; color:var(--text-secondary);" id="facesCount">0 face(s)</span>
                </div>

                <div class="metric-hero">
                    <div class="metric-label">Detected Emotion</div>
                    <div class="metric-value" id="primaryEmotion">Neutral</div>
                    <div class="metric-confidence" id="primaryConfidence">0.0% Confidence</div>
                </div>

                <div class="bars-list" id="barsList">
                    <!-- Probability bars populated dynamically -->
                </div>
            </div>
        </aside>
    </main>

    <footer>
        Deep Learning Facial Emotion Classification &bull; FER-2013 Model
    </footer>

    <!-- Hidden canvas for capturing frames -->
    <canvas id="captureCanvas" style="display:none;"></canvas>

    <script>
        const EMOTIONS = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear'];
        const COLOR_MAP = {
            'Neutral': '#c8c8c8',
            'Happiness': '#00e5ff',
            'Surprise': '#ffff00',
            'Sadness': '#ff6400',
            'Anger': '#ff1744',
            'Disgust': '#00e676',
            'Fear': '#d500f9'
        };

        const video = document.getElementById('videoElement');
        const overlayCanvas = document.getElementById('overlayCanvas');
        const overlayCtx = overlayCanvas.getContext('2d');
        const captureCanvas = document.getElementById('captureCanvas');
        const captureCtx = captureCanvas.getContext('2d');
        const barsList = document.getElementById('barsList');

        let isProcessing = false;
        let mediaStream = null;
        let processInterval = null;

        // Render initial bar elements
        EMOTIONS.forEach(emo => {
            const item = document.createElement('div');
            item.className = 'bar-item';
            item.innerHTML = `
                <div class="bar-info">
                    <span class="bar-name">${emo}</span>
                    <span class="bar-percentage" id="val-${emo}">0.0%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" id="fill-${emo}" style="background-color: ${COLOR_MAP[emo]}"></div>
                </div>
            `;
            barsList.appendChild(item);
        });

        async function startCamera() {
            try {
                if (mediaStream) stopCamera();
                mediaStream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" } 
                });
                video.srcObject = mediaStream;
                await video.play();

                document.getElementById('cameraPrompt').style.display = 'none';
                document.getElementById('stopControls').style.display = 'flex';
                document.getElementById('statusText').innerText = 'LIVE STREAMING';
                
                const syncDimensions = () => {
                    const w = video.videoWidth || 640;
                    const h = video.videoHeight || 480;
                    overlayCanvas.width = w;
                    overlayCanvas.height = h;
                    captureCanvas.width = w;
                    captureCanvas.height = h;
                };

                syncDimensions();
                video.onloadedmetadata = syncDimensions;
                video.onresize = syncDimensions;

                if (processInterval) clearInterval(processInterval);
                processInterval = setInterval(processFrame, 150);
            } catch (err) {
                console.error("Camera access error:", err);
                alert("Could not access camera: " + err.message);
            }
        }

        function stopCamera() {
            if (processInterval) {
                clearInterval(processInterval);
                processInterval = null;
            }
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }
            if (video.srcObject) {
                video.srcObject = null;
            }
            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            document.getElementById('cameraPrompt').style.display = 'flex';
            document.getElementById('stopControls').style.display = 'none';
            document.getElementById('statusText').innerText = 'CAMERA STOPPED';
            document.getElementById('facesCount').innerText = "0 face(s)";
        }

        window.addEventListener('beforeunload', stopCamera);

        async function processFrame() {
            if (isProcessing || video.paused || video.ended || !video.videoWidth) return;
            isProcessing = true;

            captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
            const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.7);

            try {
                const res = await fetch('/api/predict_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: dataUrl })
                });

                const data = await res.json();
                drawResults(data);
            } catch (err) {
                console.error("Prediction error:", err);
            } finally {
                isProcessing = false;
            }
        }

        function drawResults(data) {
            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

            const numFaces = (data.faces && data.faces.length > 0) ? data.faces.length : 0;
            document.getElementById('facesCount').innerText = numFaces > 0 ? `${numFaces} face(s) detected` : "Active stream analysis";
            document.getElementById('primaryEmotion').innerText = data.primary_emotion || "Neutral";
            document.getElementById('primaryConfidence').innerText = `${(data.confidence || 0.0).toFixed(1)}% Confidence`;

            // Always update probability bars with real neural network prediction percentages
            EMOTIONS.forEach(emo => {
                const prob = (data.probabilities && data.probabilities[emo]) ? data.probabilities[emo] : 0.0;
                document.getElementById(`val-${emo}`).innerText = `${prob.toFixed(1)}%`;
                document.getElementById(`fill-${emo}`).style.width = `${prob}%`;
            });

            // Draw bounding boxes on canvas overlay when face boxes exist
            if (data.faces && data.faces.length > 0) {
                data.faces.forEach(f => {
                    const { x, y, w, h, emotion, confidence } = f;
                    const mirroredX = overlayCanvas.width - x - w;
                    const color = COLOR_MAP[emotion] || '#00e5ff';

                    overlayCtx.strokeStyle = color;
                    overlayCtx.lineWidth = 3;
                    overlayCtx.strokeRect(mirroredX, y, w, h);

                    const text = `${emotion} (${confidence.toFixed(1)}%)`;
                    overlayCtx.font = "bold 16px Outfit, sans-serif";
                    const textWidth = overlayCtx.measureText(text).width;
                    
                    const labelY = Math.max(y - 30, 0);
                    overlayCtx.fillStyle = color;
                    overlayCtx.fillRect(mirroredX, labelY, textWidth + 16, 26);

                    overlayCtx.fillStyle = "#000000";
                    overlayCtx.fillText(text, mirroredX + 8, Math.max(y - 12, 18));
                });
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict_frame', methods=['POST'])
def predict_frame():
    if model is None or face_cascade is None:
        return jsonify({"error": "Model or face cascade resources not loaded"}), 500

    req = request.get_json()
    if not req or 'image' not in req:
        return jsonify({"error": "No image data provided"}), 400

    image_data = req['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid frame data"}), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Multi-stage detection using primary and secondary cascades
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    if len(faces_detected) == 0 and face_cascade_alt is not None:
        faces_detected = face_cascade_alt.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    if len(faces_detected) == 0:
        faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20))

    faces_res = []
    primary_emotion = "Neutral"
    top_confidence = 0.0
    probabilities = {e: 0.0 for e in EMOTIONS}

    if len(faces_detected) > 0:
        for (x, y, w, h) in faces_detected:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = np.expand_dims(roi_gray, axis=-1)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels = img_pixels.astype('float32') / 255.0

            preds = model.predict(img_pixels, verbose=0)[0]
            max_idx = int(np.argmax(preds))
            confidence = float(preds[max_idx]) * 100
            emotion = EMOTIONS[max_idx]

            faces_res.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "emotion": emotion,
                "confidence": round(confidence, 1)
            })

            if confidence > top_confidence:
                top_confidence = confidence
                primary_emotion = emotion

            for idx, e in enumerate(EMOTIONS):
                probabilities[e] = round(float(preds[idx]) * 100, 1)
    else:
        # Fallback: predict on center region of frame so analysis always returns real neural network values
        h_f, w_f = gray.shape
        crop_h, crop_w = int(h_f * 0.65), int(w_f * 0.65)
        start_y, start_x = (h_f - crop_h) // 2, (w_f - crop_w) // 2
        roi_gray = gray[start_y:start_y + crop_h, start_x:start_x + crop_w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = np.expand_dims(roi_gray, axis=-1)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels = img_pixels.astype('float32') / 255.0

        preds = model.predict(img_pixels, verbose=0)[0]
        max_idx = int(np.argmax(preds))
        top_confidence = float(preds[max_idx]) * 100
        primary_emotion = EMOTIONS[max_idx]

        for idx, e in enumerate(EMOTIONS):
            probabilities[e] = round(float(preds[idx]) * 100, 1)

    return jsonify({
        "faces": faces_res,
        "primary_emotion": primary_emotion,
        "confidence": round(top_confidence, 1),
        "probabilities": probabilities
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n========================================================")
    print("  Facial Emotion Recognition Web Application")
    print(f"  Open in Web Browser: http://localhost:{port}")
    print("========================================================\n", flush=True)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
