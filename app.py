from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cv2
import os
from inference_models import predict_all
import threading
import json

app = Flask(__name__)

face_detector = YOLO("yolov8n-face.pt")
gender_model = YOLO("y8n_agegender_gender20/weights/best.pt")
emotion_model = YOLO("y8n_emotion19/weights/best.pt")

script_dir = os.path.dirname(os.path.abspath(__file__))
haar_path = os.path.join(script_dir, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(haar_path)

IMG_SIZE = 224
TARGET_SIZE = (224, 224)

cap = None
lock = threading.Lock()
current_frame = None

def capture_frames():
    global cap, current_frame
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if ret:
            with lock:
                current_frame = frame.copy()

def get_frame():
    with lock:
        if current_frame is not None:
            return current_frame.copy()
    return None

def preprocess_face(frame, box):
    x, y, w, h = box
    h_frame, w_frame, _ = frame.shape
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_frame - x)
    h = min(h, h_frame - y)

    face = frame[y:y + h, x:x + w]
    if face.size == 0:
        return None

    face = cv2.resize(face, TARGET_SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    return face

def generate_yolo_frames():
    while True:
        frame = get_frame()
        if frame is None:
            continue

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

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_cnn_frames():
    while True:
        frame = get_frame()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_arr = preprocess_face(frame, (x, y, w, h))
            if face_arr is None:
                continue

            age, gender, expr = predict_all(face_arr)

            label = f"{age}, {gender}, {expr}"

            text_x, text_y = x, max(0, y - 10)
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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


try:
    with open('votes.json', 'r') as f:
        votes = json.load(f)
except FileNotFoundError:
    votes = {"yolo_votes": 0, "cnn_votes": 0}

@app.route('/vote', methods=['POST'])
def vote():
    data = request.get_json()
    model = data.get('model')

    model += "_votes"
    
    if model in votes:
        votes[model] += 1

        with open('votes.json', 'w') as f:
            json.dump(votes, f)

    return jsonify({
        'yolo_votes': votes['yolo_votes'],
        'cnn_votes': votes['cnn_votes']
    })

@app.route('/votes', methods=['GET'])
def get_votes():
    return jsonify(votes)

if __name__ == '__main__':
    camera_thread = threading.Thread(target=capture_frames, daemon=True)
    camera_thread.start()
    
    app.run(debug=True, threaded=True, use_reloader=False)