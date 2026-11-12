from flask import Flask, Response
import cv2
import numpy as np
import os
import torch

os.environ['OMP_NUM_THREADS'] = '1'

app = Flask(__name__)

print("Loading YOLO model...")

# Load YOLOv8 model with torch (no multiprocessing)
from ultralytics import YOLO

# Disable all multiprocessing in torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

model = YOLO('yolov8n.pt')
model.to('cpu')

print("✓ YOLO loaded")

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("⚠️ Could not open camera")
        return
    
    print("✓ Camera opened")
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # Process every 2nd frame to reduce load
        if frame_count % 2 == 0:
            try:
                # Run YOLO inference with minimal settings
                results = model(frame, verbose=False, device='cpu', half=False)
                
                # Draw results
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            label = f"{model.names[cls]}: {conf:.2f}"
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                pass  # Skip frame if error
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()
    print("Camera released")

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>Object Detection System</title>
        <style>
            body { background: #1a1a2e; color: white; font-family: Arial; text-align: center; margin: 0; padding: 20px; }
            h1 { color: #00ff00; margin-top: 20px; }
            img { border: 3px solid #00ff00; border-radius: 10px; max-width: 90%; }
            .status { margin-top: 10px; color: #00ff00; font-size: 14px; }
            .warning { margin-top: 20px; color: #ffaa00; font-size: 12px; }
        </style>
    </head>
    <body>
        <h1>🎥 Real-Time Object Detection</h1>
        <img src="/video_feed" id="video" onload="setTimeout(() => { console.log('Frame loaded'); }, 100)">
        <div class="status">✓ System Running | YOLOv8 | Object Detection Active</div>
        <div class="warning">Detecting: person, cell phone, laptop, and 80+ other objects</div>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Object Detection Server")
    print("="*50)
    print("📱 Open http://localhost:5000 in your browser")
    print("🎥 Camera should activate automatically")
    print("⚠️  If video doesn't appear, check browser permissions")
    print("Press Ctrl+C to stop")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False, use_reloader=False)
