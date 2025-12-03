from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

app = Flask(__name__)

face_detector = YOLO("yolov8n-face.pt")
gender_model = YOLO("y8n_agegender_gender20/weights/best.pt")
emotion_model = YOLO("y8n_emotion19/weights/best.pt")

IMG_SIZE = 224

def generate_yolo_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = face_detector(frame, stream=True)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

                g_result = gender_model(face_resized)
                g_probs = g_result[0].probs 
                gender_idx = g_probs.top1
                gender_label = g_result[0].names[gender_idx]

                e_res = emotion_model(face_resized)
                e_probs = e_res[0].probs
                e_idx = e_probs.top1
                e_label = e_res[0].names[e_idx]

                label = f"{gender_label} | {e_label}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def generate_cnn_frames():
    return


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/yolo_feed')
def yolo_feed():
    return Response(generate_yolo_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cnn_feed')
def cnn_feed():
    return Response(generate_cnn_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)