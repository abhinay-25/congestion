# Intelligent Traffic System

This project implements an intelligent traffic monitoring system using computer vision and deep learning. It uses YOLOv8 for vehicle detection and provides real-time traffic density analysis.

## Features

- Real-time vehicle detection using YOLOv8
- Traffic density estimation
- Web interface for monitoring
- Vehicle counting and classification

## Setup Instructions

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. The system will automatically download the YOLOv8 model on first run.

## Running the System

1. Start the application:
```bash
python traffic_system.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. The system will use your default webcam. To use a different video source:
   - Edit `traffic_system.py`
   - Change `cv2.VideoCapture(0)` to your desired source (e.g., video file path)

## System Requirements

- Python 3.8 or higher
- Webcam or video source
- Minimum 4GB RAM
- CUDA-capable GPU (recommended for better performance)

## Notes

- The system updates traffic density every 5 seconds
- Vehicle count thresholds can be adjusted in the `TrafficSystem` class
- For better performance, consider using a more powerful YOLO model variant 