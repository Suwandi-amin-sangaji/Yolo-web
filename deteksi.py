import torch
import cv2

# Load model
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')

cap = cv2.VideoCapture(0)  # Gunakan webcam default

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detected_labels = [label['name'] for label in results.pandas().xyxy[0].to_dict(orient="records")]
    print(f"Detected labels: {detected_labels}")

    # Render hasil deteksi
    res_plotted = results.render()[0]

    # Tampilkan frame dengan hasil deteksi
    cv2.imshow('YOLOv5 Webcam', res_plotted)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
