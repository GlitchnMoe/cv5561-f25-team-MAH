from ultralytics import YOLO
import cv2
# import torch

face_detector = YOLO("yolov8n.pt") # this should proabbly be face

gender_model = YOLO("runs_emotion/yolo_runs15/weights/best.pt")
emotion_model = YOLO("y8n_emotion19/weights/best.pt")

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

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

            g_result = gender_model(face_resized)
            g_probs = g_result[0].probs
            g_idx = g_probs.top1
            gender_label = g_result[0].names[g_idx]



            e_res = emotion_model(face_resized)
            e_probs = e_res[0].probs
            e_idx = e_probs.top1
            e_label = e_res[0].names[e_idx]

            label = f"{gender_label} | {e_label}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imshow("Gender + Emotion Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
