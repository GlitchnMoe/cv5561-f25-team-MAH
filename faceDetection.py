from ultralytics import YOLO
import cv2
import torch

face_detector = YOLO("yolov8n.pt")

gender_model = YOLO("y8n_agegender_gender3/weights/best.pt")
emotion_model = YOLO("y8n_emotion/y8n_emotion/weights/best.pt")

IMG_SIZE = 224

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = face_detector(frame, stream=True)
    

    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            if r.names[cls] != "person":
                continue

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

    cv2.imshow("Gender + Emotion Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
