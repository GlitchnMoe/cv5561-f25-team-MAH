# one script for processing, one for training, one for prediction
import dataProcessing
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    model.train(
        data="./UTKFace_YOLO/data.yaml",
        device=0,
        epochs=50,
        imgsz=256,
        batch=16,
        project="runs/UTKFace",
        name="age_gender"
    )
    


# import cv2
# from ultralytics import YOLO

# # Load YOLOv8 model
# model = YOLO("yolov8n.pt")  # nano model is fast for real-time

# # Open default webcam
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open webcam")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run YOLO prediction on the frame
#     results = model(frame)  # returns a Results object

#     # Draw predictions on the frame
#     annotated_frame = results[0].plot()  # plot boxes and labels on frame

#     # Show the frame
#     cv2.imshow("YOLOv8 Webcam", annotated_frame)

#     # Exit on 'q' key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
