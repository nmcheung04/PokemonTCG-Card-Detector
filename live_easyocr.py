import easyocr
import cv2
import numpy as np
import time
import threading
from collections import deque

class LiveEasyOCR:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.reader = None
        self.is_running = False
        self.current_frame = None
        self.ocr_results = []
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0
        self.show_fps = True
        self.show_confidence = True
        self.min_confidence = 0.5
        self.ocr_interval = 0.5
        self.last_ocr_time = 0
        # ROI settings (centered rectangle)
        self.roi_width = 500
        self.roi_height = 300
        print("Initializing EasyOCR... (this may take a moment)")
        self.reader = easyocr.Reader(['en'])
        print("EasyOCR initialized successfully!")

    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print("Camera started successfully!")
        return True

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera stopped.")

    def get_roi(self, frame):
        h, w = frame.shape[:2]
        # Card aspect ratio (width:height)
        card_aspect = 0.714  # 2.5/3.5
        # ROI height: 80% of frame height (10% margin top/bottom)
        roi_height = int(0.8 * h)
        roi_width = int(card_aspect * roi_height)
        # If ROI width is too wide for the frame, shrink both
        if roi_width > w:
            roi_width = int(0.8 * w)
            roi_height = int(roi_width / card_aspect)
        x1 = w // 2 - roi_width // 2
        y1 = h // 2 - roi_height // 2
        x2 = x1 + roi_width
        y2 = y1 + roi_height
        return (x1, y1, x2, y2, roi_width, roi_height), frame[y1:y2, x1:x2]

    def run_ocr_on_frame(self, frame):
        (x1, y1, x2, y2, roi_width, roi_height), roi = self.get_roi(frame)
        # Only use the top 20% of the ROI for OCR
        name_height = int(0.2 * roi_height)
        name_roi = roi[0:name_height, :]
        try:
            results = self.reader.readtext(name_roi)
            filtered_results = []
            for bbox, text, conf in results:
                if conf >= self.min_confidence:
                    # Correctly shift bbox coordinates to full frame
                    shifted_bbox = [[pt[0] + x1, pt[1] + y1] for pt in bbox]
                    filtered_results.append((shifted_bbox, text, conf))
            return filtered_results
        except Exception as e:
            print(f"OCR error: {e}")
            return []

    def process_frame(self, frame):
        display_frame = frame.copy()
        (x1, y1, x2, y2, roi_width, roi_height), _ = self.get_roi(frame)
        # Draw ROI rectangle
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Draw horizontal line at 20% from top inside ROI
        line_y = y1 + int(0.2 * roi_height)
        cv2.line(display_frame, (x1, line_y), (x2, line_y), (255, 0, 0), 2)
        # Draw OCR results
        for bbox, text, confidence in self.ocr_results:
            bbox = np.array(bbox, dtype=np.int32)
            cv2.polylines(display_frame, [bbox], True, (0, 255, 0), 2)
            x, y = bbox[0]
            label = text
            if self.show_confidence:
                label += f" ({confidence:.2f})"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(display_frame, (int(x), int(y) - text_height - 10), (int(x) + text_width, int(y)), (0, 255, 0), -1)
            cv2.putText(display_frame, label, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        if self.show_fps:
            cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'q' to quit, 's' to save frame", (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return display_frame

    def save_current_frame(self):
        if self.current_frame is not None:
            timestamp = int(time.time())
            filename = f"ocr_capture_{timestamp}.jpg"
            processed_frame = self.process_frame(self.current_frame)
            cv2.imwrite(filename, processed_frame)
            print(f"Frame saved as: {filename}")

    def run(self):
        if not self.start_camera():
            return
        self.is_running = True
        print("Live OCR started! Press 'q' to quit, 's' to save frame")
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame")
                break
            self.current_frame = frame
            current_time = time.time()
            if current_time - self.last_ocr_time >= self.ocr_interval:
                self.ocr_results = self.run_ocr_on_frame(frame)
                self.last_ocr_time = current_time
            display_frame = self.process_frame(frame)
            self.fps_counter += 1
            if current_time - self.fps_start_time >= 1.0:
                self.fps = self.fps_counter / (current_time - self.fps_start_time)
                self.fps_counter = 0
                self.fps_start_time = current_time
            cv2.imshow('Live EasyOCR', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_current_frame()
            elif key == ord('c'):
                self.ocr_results = []
                print("OCR results cleared")
            elif key == ord('f'):
                self.show_fps = not self.show_fps
                print(f"FPS display: {'ON' if self.show_fps else 'OFF'}")
            elif key == ord('d'):
                self.show_confidence = not self.show_confidence
                print(f"Confidence display: {'ON' if self.show_confidence else 'OFF'}")
        self.stop_camera()

def main():
    print("Live EasyOCR Card Detection (ROI mode)")
    print("=" * 40)
    print("Controls:")
    print("  q - Quit")
    print("  s - Save current frame")
    print("  c - Clear OCR results")
    print("  f - Toggle FPS display")
    print("  d - Toggle confidence display")
    print("=" * 40)
    live_ocr = LiveEasyOCR(camera_index=0)
    try:
        live_ocr.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        live_ocr.stop_camera()

if __name__ == "__main__":
    main() 