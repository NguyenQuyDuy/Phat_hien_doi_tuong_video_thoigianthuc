import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from datetime import datetime
import time

# Khởi tạo mô hình YOLO
model = YOLO('yolov5s.pt')  


# Biến toàn cục để lưu frame hiện tại
current_frame = None

# Hàm chụp màn hình
def capture_screenshot():
    global current_frame
    if current_frame is not None:
        # Tạo thư mục lưu ảnh
        save_dir = "fileluuanh"
        os.makedirs(save_dir, exist_ok=True)
        
        # Tạo tên file với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"fileluuanh_{timestamp}.jpg")

        # vẽ khung và ID lên ảnh chụp
        frame_with_boxes = current_frame.copy()  # Sao chép và xử lí khung hình
        results = model(frame_with_boxes)
        person_count = 1

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                label = result.names[int(box.cls[0])]

                if label == "person":
                    label_text = f'Person {person_count} {confidence:.2f}'
                    person_count += 1
                else:
                    label_text = f'{label} {confidence:.2f}'

                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_with_boxes, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Lưu ảnh với các khung và ID đối tượng
        cv2.imwrite(filename, frame_with_boxes)
        messagebox.showinfo("Thông báo", f"Ảnh đã được lưu thành công!\nĐường dẫn: {filename}")
    else:
        messagebox.showwarning("Cảnh báo", "Không có khung hình nào để chụp!")

# Hàm phát hiện đối tượng bằng webcam
def detect_objects_with_webcam():
    global current_frame
    cap = cv2.VideoCapture(0)  # Mở webcam
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không mở được webcam!")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Lấy FPS từ webcam
    delay = max(1, int(1000 / fps))  # Tính thời gian chờ phù hợp
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))  # Tối ưu khung hình
        current_frame = frame.copy()  # Lưu frame hiện tại để chụp màn hình
        person_count = 1
        results = model(frame)
# Vẽ khung và thông tin đối tượng
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                label = result.names[int(box.cls[0])]
# vẽ khung và nhãn
                if label == "person":
                    label_text = f'Person {person_count} {confidence:.2f}'
                    person_count += 1
                else:
                    label_text = f'{label} {confidence:.2f}'

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Tính và hiển thị FPS
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time
        start_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Object Detection with Webcam', frame)
        if cv2.waitKey(delay) & 0xFF == ord('0'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Hàm phát hiện đối tượng trong video
def detect_objects_in_video():
    global current_frame
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if not video_path:
        messagebox.showinfo("Thông báo", "Không có video nào được chọn!")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không mở được video!")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Lấy FPS từ video
    delay = max(1, int(1000 / fps))  # Tính thời gian chờ phù hợp

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (700, 600))  # Tối ưu khung hình
        current_frame = frame.copy()  # Lưu frame hiện tại để chụp màn hình
        person_count = 1
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                label = result.names[int(box.cls[0])]

                if label == "person":
                    label_text = f'Person {person_count} {confidence:.2f}'
                    person_count += 1
                else:
                    label_text = f'{label} {confidence:.2f}'

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Object Detection in Video', frame)
        key = cv2.waitKey(delay)
        if key & 0xFF == ord('1') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Giao diện 
root = tk.Tk()
root.title("Hệ thống phát hiện và nhận diện đối tượng")
root.geometry("1000x700")
root.configure(bg="#f2f2f2")

title_label = tk.Label(root, text="Hệ thống phát hiện và nhận diện đối tượng", font=("Arial", 20, "bold"), bg="#f2f2f2", fg="#333")
title_label.pack(pady=20)

button_frame = tk.Frame(root, bg="#f2f2f2")
button_frame.pack(pady=40)

button_webcam = tk.Button(
    button_frame, text="Nhận diện Đối tượng bằng Webcam", command=detect_objects_with_webcam,
    padx=20, pady=10, font=("Arial", 12), bg="#007bff", fg="white", relief="solid"
)
button_webcam.grid(row=0, column=0, padx=10, pady=10)

button_video = tk.Button(
    button_frame, text="Chọn Video để Nhận diện Đối tượng", command=detect_objects_in_video,
    padx=20, pady=10, font=("Arial", 12), bg="#28a745", fg="white", relief="solid"
)
button_video.grid(row=0, column=1, padx=10, pady=10)

# Nút chụp màn hình
button_capture = tk.Button(
    root, text="Chụp màn hình", command=capture_screenshot,
    padx=20, pady=10, font=("Arial", 12), bg="#ffc107", fg="white", relief="solid"
)
button_capture.pack(pady=20)

button_exit = tk.Button(
    root, text="Thoát", command=root.quit,
    padx=20, pady=10, font=("Arial", 12), bg="#dc3545", fg="white", relief="solid"
)
button_exit.pack(pady=20)

root.mainloop()
