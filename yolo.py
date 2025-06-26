import cv2
import numpy as np
from ultralytics import YOLO
import csv
import os

# === ПУТИ ===
VIDEO_PATH = '/home/se/Загрузки/vid2205/cvtest.avi'
CSV_PATH = 'detections_log.csv'

# === ИНИЦИАЛИЗАЦИЯ ===
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(VIDEO_PATH)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

# === ROI (шлагбаумы) как многоугольники ===
roi_left = np.array([[50, 450], [490, 450], [490, 1000], [50, 1000]], dtype=np.int32)
roi_right = np.array([[1000, 450], [1480, 450], [1480, 1000], [1000, 1000]], dtype=np.int32)

# === CSV-лог ===
with open(CSV_PATH, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp_sec", "track_id", "label", "barrier"])

    # === ОСНОВНОЙ ЦИКЛ ===
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        timestamp = round(frame_num / frame_rate, 2)

        # Трекинг и детекция
        results = model.track(frame, persist=True, verbose=False)[0]
        annotated_frame = frame.copy()

        left_detected = set()
        right_detected = set()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            name = model.names[cls_id]
            if name not in ['car', 'truck', 'bus', 'motorcycle']:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            point = (cx, cy)
            barrier = None

            if cv2.pointPolygonTest(roi_left, point, False) >= 0:
                barrier = 'left'
                left_detected.add((track_id, name))
            elif cv2.pointPolygonTest(roi_right, point, False) >= 0:
                barrier = 'right'
                right_detected.add((track_id, name))

            if barrier:
                writer.writerow([timestamp, track_id, name, barrier])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{name} [{barrier}]", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Визуальные зоны
        cv2.polylines(annotated_frame, [roi_left], isClosed=True, color=(255, 255, 0), thickness=2)
        cv2.polylines(annotated_frame, [roi_right], isClosed=True, color=(255, 255, 0), thickness=2)

        # Статус
        if left_detected or right_detected:
            status = "Обнаружен транспорт"
            color = (0, 255, 0)
        else:
            status = "Транспорт не обнаружен"
            color = (0, 0, 255)

        cv2.putText(annotated_frame, status, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Масштабирование окна
        resized = cv2.resize(annotated_frame, (1280, 720))
        cv2.imshow("Detection", resized)

        if cv2.waitKey(1) == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Детекция завершена. Результаты сохранены в файл: {CSV_PATH}")

