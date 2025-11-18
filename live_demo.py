import cv2
import os
from inference_models import predict_all

# =========================
# Config
# =========================
TARGET_SIZE = (224, 224)  # change if your friend's model uses different size


# =========================
# Preprocessing
# =========================
def preprocess_face(frame, box):
    x, y, w, h = box
    h_frame, w_frame, _ = frame.shape
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_frame - x)
    h = min(h, h_frame - y)

    face = frame[y:y + h, x:x + w]       # BGR crop
    if face.size == 0:
        return None

    face = cv2.resize(face, TARGET_SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # NO division by 255 here – let torchvision handle it
    # face is uint8 RGB (H, W, 3)
    return face

# =========================
# Dummy model (replace later)
# =========================
def dummy_predict(face_arr):
    """
    face_arr: (H, W, 3) float32, [0, 1]
    For now, just return fake values.
    Later, replace with real model call.
    """
    _ = face_arr  # Acknowledge parameter (unused in dummy implementation)
    age = 25
    gender = "Male"
    expr = "Happy"
    return age, gender, expr

# =========================
# Main loop
# =========================
def main():
    # 1. Open webcam
    cap = cv2.VideoCapture(0)  # 0 = default camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # 2. Load Haar Cascade for face detection
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    haar_path = os.path.join(script_dir, "haarcascade_frontalface_default.xml")

    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(haar_path)
    if face_cascade.empty():
        print("Error: Could not load Haar Cascade.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Optionally resize frame for speed (uncomment if needed)
        # frame = cv2.resize(frame, (640, 480))

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3. Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # 4. For each face, preprocess + predict + draw
        for (x, y, w, h) in faces:
            # Draw bounding box first
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Preprocess face for model
            face_arr = preprocess_face(frame, (x, y, w, h))
            if face_arr is None:
                continue

            # TODO: replace this with real model prediction later
            age, gender, expr = predict_all(face_arr)

            # 5. Create label text
            label = f"{age}, {gender}, {expr}"

            # 6. Put label above the box
            text_x, text_y = x, max(0, y - 10)
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        # 7. Show result
        cv2.imshow("Live Age/Gender/Expression Demo (Mock)", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()