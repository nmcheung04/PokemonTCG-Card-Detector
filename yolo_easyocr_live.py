import cv2
import numpy as np
import time
import easyocr
from ultralytics import YOLO

def preprocess_for_ocr(roi):
    # Only upscale, keep color
    return cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

def get_centered_card_roi(frame, aspect_ratio=0.714, margin=0.05):
    h, w = frame.shape[:2]
    roi_height = int((1 - 2 * margin) * h)
    roi_width = int(aspect_ratio * roi_height)
    if roi_width > w:
        roi_width = int((1 - 2 * margin) * w)
        roi_height = int(roi_width / aspect_ratio)
    x1 = w // 2 - roi_width // 2
    y1 = int(margin * h)
    x2 = x1 + roi_width
    y2 = y1 + roi_height
    return (x1, y1, x2, y2, roi_width, roi_height), frame[y1:y2, x1:x2]

def main():
    print("Loading YOLOv8 model...")
    model = YOLO('best.pt')
    print("Model loaded successfully!")
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'])
    print("EasyOCR initialized!")
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("Webcam initialized. Press 'q' to quit, 's' to save frame.")
    frame_count = 0
    start_time = time.time()
    min_conf = 0.3
    card_aspect = 0.714
    ocr_name_region = 0.2  # Top 20% of ROI
    yolo_interval = 0.5  # seconds
    last_yolo_time = 0
    last_yolo_boxes = []
    last_ocr_results = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: Could not read frame, retrying...")
            time.sleep(0.1)
            continue
        # 1. Make a copy for display overlays
        display_frame = frame.copy()
        # 2. Get blue box ROI (centered, card aspect, 5% margin)
        (rx1, ry1, rx2, ry2, roi_w, roi_h), roi = get_centered_card_roi(frame, aspect_ratio=card_aspect, margin=0.05)
        # 3. Run YOLO and OCR only on the clean ROI (no overlays)
        current_time = time.time()
        run_yolo = (current_time - last_yolo_time) >= yolo_interval
        if run_yolo:
            try:
                yolo_results = model(roi, conf=min_conf)
                last_yolo_boxes = []
                for result in yolo_results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            conf = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = "card"
                            fx1, fy1, fx2, fy2 = x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1
                            last_yolo_boxes.append((fx1, fy1, fx2, fy2, conf, class_name))
                # OCR only on top 20% of ROI
                name_h = int(0.2 * roi_h)
                name_roi = roi[0:name_h, :]
                # Preprocess for small text (only upscale)
                name_roi_pre = preprocess_for_ocr(name_roi)
                last_ocr_results = []
                try:
                    ocr_results = reader.readtext(name_roi_pre)
                except Exception as e:
                    print(f"OCR error: {e}")
                    ocr_results = []
                for bbox, text, ocr_conf in ocr_results:
                    if ocr_conf < 0.5:
                        continue
                    mapped_bbox = [[int(pt[0]+rx1), int(pt[1]+ry1)] for pt in bbox]
                    last_ocr_results.append((mapped_bbox, text, ocr_conf))
                last_yolo_time = current_time
            except Exception as e:
                print(f"Error during YOLO inference: {e}")
        # 4. Draw overlays (blue box, horizontal line) on display_frame only
        cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        line_y = ry1 + int(0.2 * roi_h)
        cv2.line(display_frame, (rx1, line_y), (rx2, line_y), (255, 0, 0), 2)
        # 5. Draw last YOLO detections on display_frame
        for fx1, fy1, fx2, fy2, conf, class_name in last_yolo_boxes:
            cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(display_frame, (fx1, fy1 - label_size[1] - 10), (fx1 + label_size[0], fy1), (0, 255, 0), -1)
            cv2.putText(display_frame, label, (fx1, fy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        # 6. Draw last OCR results on display_frame
        for mapped_bbox, text, ocr_conf in last_ocr_results:
            cv2.polylines(display_frame, [np.array(mapped_bbox, dtype=np.int32)], True, (0, 255, 255), 2)
            mx, my = mapped_bbox[0]
            ocr_label = f"{text} ({ocr_conf:.2f})"
            (tw, th), _ = cv2.getTextSize(ocr_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(display_frame, (mx, my - th - 10), (mx + tw, my), (0, 255, 255), -1)
            cv2.putText(display_frame, ocr_label, (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        # 7. FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if frame_count % 30 == 0:
            print(f"FPS: {fps:.1f}")
        cv2.imshow('YOLO (ROI) + EasyOCR Card Detector - Press Q to Quit', display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'card_detection_{int(time.time())}.jpg', display_frame)
            print("Frame saved!")
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released. Goodbye!")

if __name__ == "__main__":
    main() 