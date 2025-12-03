from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO

face_detector = YOLO("yolov8n.pt")

# Initialize models
gender_model = YOLO("y8n_agegender_gender3/weights/best.pt")
emotion_model = YOLO("y8n_emotion/y8n_emotion/weights/best.pt")
IMG_SIZE = 224

cap = cv2.VideoCapture("test_video.mp4")

app = Flask(__name__)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        ret, frame = cap.read()
    
        results = face_detector(frame, stream=True)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

                g_pred = gender_model(face_resized)[0]
                gender_label = "Unknown"
                if len(g_pred.boxes) > 0:
                    best = g_pred.boxes[0]
                    g_cls = int(best.cls)
                    g_conf = float(best.conf)
                    gender_label = f"{gender_model.names[g_cls]} {g_conf:.2f}"


                e_pred = emotion_model(face_resized)[0]
                emotion_label = "Unknown"
                if len(e_pred.boxes) > 0:
                    best = e_pred.boxes[0]
                    e_cls = int(best.cls)
                    e_conf = float(best.conf)
                    emotion_label = f"{emotion_model.names[e_cls]} {e_conf:.2f}"

                label = f"{gender_label} | {emotion_label}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

as multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
