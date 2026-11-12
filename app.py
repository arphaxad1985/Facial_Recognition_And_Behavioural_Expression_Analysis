import os
import sys

# CRITICAL FIX: Disable all multiprocessing before anything else
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['ULTRALYTICS_NO_MULTIPROCESSING'] = '1'

# Set torch to single thread if available
try:
    import torch
    torch.set_num_threads(1)
except:
    pass

# Now import everything else
from flask import Flask, render_template, Response, jsonify
import cv2
import time
import threading
from datetime import datetime

# Import YOLO
from ultralytics import YOLO

# Try to import FER, but handle gracefully
try:
    from fer import FER
    emotion_detector = FER()
    print("✓ Emotion detection enabled")
except Exception as e:
    print(f"⚠️ Emotion detection disabled: {e}")
    emotion_detector = None

# Initialize Flask
app = Flask(__name__)

# Load YOLO with CPU only
print("Loading YOLO...")
model = YOLO('yolov8n.pt')
model.to('cpu')
print("✓ YOLO loaded")

# Camera
camera = None
camera_lock = threading.Lock()

def get_camera():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return camera

def generate_frames():
    cap = get_camera()
    frame_count = 0
    
    while True:
        try:
            success, frame = cap.read()
            if not success:
                continue
            
            frame_count += 1
            
            # Process detection every 3 frames (smoother)
            if frame_count % 3 == 0:
                # YOLO detection
                results = model(frame, verbose=False, device='cpu')
                
                # Draw boxes
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            label = f"{model.names[cls]}: {conf:.2f}"
                            
                            # Red box for phones (class 67 is cell phone)
                            color = (0, 0, 255) if model.names[cls] == 'cell phone' else (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1-5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Emotion detection (if available)
                if emotion_detector and frame_count % 6 == 0:
                    try:
                        emotions = emotion_detector.detect_emotions(frame)
                        if emotions:
                            top_emotion = max(emotions[0]['emotions'].items(), key=lambda x: x[1])
                            cv2.putText(frame, f"Emotion: {top_emotion[0]}", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    except:
                        pass
            
            # Add timestamp
            cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            print(f"Frame error: {e}")
            continue

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ProctorAI - Exam Monitoring</title>
        <style>
            body { font-family: Arial; background: #1a1a2e; color: white; margin: 0; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: #00ff00; text-align: center; }
            .video-container { text-align: center; margin: 20px 0; }
            video, img { border: 2px solid #00ff00; border-radius: 10px; max-width: 100%; }
            .status { text-align: center; margin: 10px; color: #00ff00; }
            .info { background: #0f0f1a; padding: 15px; border-radius: 10px; margin-top: 20px; }
            .green { color: #00ff00; }
            .red { color: #ff4444; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎓 ProctorAI - Exam Monitoring System</h1>
            <div class="video-container">
                <img src="/video_feed" width="800">
            </div>
            <div class="status">
                <span class="green">● System Active</span> | Monitoring: Objects & Emotions
            </div>
            <div class="info">
                <strong>📋 Detection Features:</strong><br>
                • 📱 Cell Phone Detection (red boxes)<br>
                • 👤 Person Detection (green boxes)<br>
                • 😊 Emotion Recognition (if available)<br>
                • 📸 Live Monitoring Feed
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎓 ProctorAI - Exam Monitoring System")
    print("="*50)
    print(f"📍 Open: http://localhost:5000")
    print(f"🔴 Press Ctrl+C to stop")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
