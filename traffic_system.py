import cv2
import numpy as np
from ultralytics import YOLO
import time
from flask import Flask, render_template, Response, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
uploaded_video_path = None

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model

class TrafficSystem:
    def __init__(self):
        self.vehicle_count = 0
        self.last_update = time.time()
        self.update_interval = 5  # seconds
        self.traffic_density = "Low"
        
    def process_frame(self, frame):
        results = model(frame)
        vehicle_classes = [2, 3, 5, 7]
        current_count = 0
        congestion_length = 0
        frame_height = frame.shape[0]
        min_y = frame_height
        max_y = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls) in vehicle_classes:
                    current_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    min_y = min(min_y, y1)
                    max_y = max(max_y, y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if current_count > 0:
            congestion_length = max_y - min_y
        if time.time() - self.last_update >= self.update_interval:
            self.vehicle_count = current_count
            self.last_update = time.time()
            if self.vehicle_count > 20:
                self.traffic_density = "High"
            elif self.vehicle_count > 10:
                self.traffic_density = "Medium"
            else:
                self.traffic_density = "Low"
        # Draw results on frame
        annotated_frame = frame
        cv2.putText(annotated_frame, f"Vehicles: {self.vehicle_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Traffic: {self.traffic_density}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Congestion Length: {congestion_length}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return annotated_frame

# Initialize traffic system
traffic_system = TrafficSystem()

def generate_frames():
    global uploaded_video_path
    # Use uploaded video file if available, else use webcam
    video_source = uploaded_video_path if uploaded_video_path else 0
    cap = cv2.VideoCapture(video_source)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            processed_frame = traffic_system.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global uploaded_video_path
    if 'video_file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['video_file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        uploaded_video_path = file_path
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)