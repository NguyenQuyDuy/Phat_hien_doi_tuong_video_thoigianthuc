import cv2
import time
from ultralytics import YOLO

# Khởi tạo mô hình YOLO
model = YOLO('yolov5s.pt')

# Mở webcam
cap = cv2.VideoCapture(0)

# Biến đo FPS
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện đối tượng
    results = model(frame)
    frame_count += 1

    # Hiển thị kết quả
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Real-Time Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('0'):
        break

# Tính FPS
end_time = time.time()
fps = frame_count / (end_time - start_time)
print(f"FPS: {fps:.2f}")

cap.release()
cv2.destroyAllWindows()
