import cv2
import numpy as np
from ultralytics import YOLO
import time
from flask import Flask, render_template, Response
import yt_dlp
import firebase_admin
from firebase_admin import credentials, db

app = Flask(__name__)

# List of YouTube URLs for 4 lanes
LANE_URLS = [
    "https://www.youtube.com/watch?v=5_XSYlAfJZM&list=PLcQZGj9lFR7y5WikozDSrdk6UCtAnM9mB&index=7",  # Lane 1
    "https://www.youtube.com/watch?v=czcqVgHG3y0",  # Lane 2
    "https://www.youtube.com/watch?v=E8LsKcVpL5A",  # Lane 3
    "https://www.youtube.com/watch?v=DnUFAShZKus"  # Lane 4
]

# Initialize YOLOv8 model (vehicle detection)
model = YOLO('yolov8n.pt')  # Nano model for speed

# Initialize Firebase only once
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase_key.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://hackathonproject-8b9db-default-rtdb.asia-southeast1.firebasedatabase.app/'
        })

init_firebase()

def get_youtube_stream(youtube_url):
    ydl_opts = {
        'quiet': True,
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        # Find all video+audio HLS streams
        hls_streams = [
            f for f in info['formats']
            if f['protocol'] == 'm3u8_native'
            and f.get('vcodec', 'none') != 'none'
            and f.get('acodec', 'none') != 'none'
            and f.get('height') is not None
        ]
        if not hls_streams:
            raise Exception("No HLS video+audio streams found for this video.")
        # Pick the highest resolution stream
        best_stream = max(hls_streams, key=lambda f: f['height'])
        print(f"Selected stream: {best_stream['format_id']} ({best_stream['height']}p) for URL {youtube_url}")
        return cv2.VideoCapture(best_stream['url'])

class TrafficSystem:
    def __init__(self, lane_id=1):
        self.vehicle_count = 0
        self.last_update = time.time()
        self.update_interval = 5  # seconds
        self.traffic_density = "Low"
        self.lane_id = lane_id
    
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
            # Send live data to Firebase for this lane
            data = {
                'vehicle_count': self.vehicle_count,
                'traffic_density': self.traffic_density,
                'congestion_length': congestion_length,
                'timestamp': int(time.time())
            }
            db.reference(f'traffic_data/lane{self.lane_id}').set(data)
        annotated_frame = frame
        cv2.putText(annotated_frame, f"Lane {self.lane_id} Vehicles: {self.vehicle_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Traffic: {self.traffic_density}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Congestion Length: {congestion_length}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return annotated_frame

lane_systems = []
lane_caps = []
for idx, url in enumerate(LANE_URLS):
    try:
        cap = get_youtube_stream(url)
        if not cap.isOpened():
            print(f"ERROR: Could not open video stream for Lane {idx+1} ({url})")
        else:
            print(f"Lane {idx+1} stream opened successfully.")
        lane_caps.append(cap)
        lane_systems.append(TrafficSystem(lane_id=idx+1))
    except Exception as e:
        print(f"Exception initializing Lane {idx+1} ({url}): {e}")
        lane_caps.append(None)
        lane_systems.append(TrafficSystem(lane_id=idx+1))

def generate_frames(lane_idx):
    cap = lane_caps[lane_idx]
    system = lane_systems[lane_idx]
    if cap is None or not cap.isOpened():
        placeholder = np.zeros((320, 480, 3), dtype=np.uint8)
        cv2.putText(placeholder, f"Lane {lane_idx+1} unavailable", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        ret, buffer = cv2.imencode('.jpg', placeholder)
        frame = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            placeholder = np.zeros((320, 480, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Lane {lane_idx+1} unavailable", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            break
        else:
            processed_frame = system.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<int:lane_id>')
def video_feed(lane_id):
    # lane_id is 1-based
    return Response(generate_frames(lane_id-1), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)