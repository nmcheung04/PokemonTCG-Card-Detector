import cv2
import numpy as np
from ultralytics import YOLO
import time

def main():
    # Load the trained YOLOv8 model
    print("Loading YOLOv8 model...")
    model = YOLO('best.pt')
    print("Model loaded successfully!")
    
    # Initialize webcam - try different indices if needed
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)  # Try index 0 first
    
    # If index 0 doesn't work, try index 1
    if not cap.isOpened():
        print("Trying webcam index 1...")
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check your webcam connection.")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Webcam initialized successfully!")
    print("Press 'q' to quit.")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("Warning: Could not read frame, retrying...")
            time.sleep(0.1)
            continue
        
        # Run YOLOv8 inference on the frame
        try:
            results = model(frame, conf=0.3)  # Lower confidence threshold for more detections
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get confidence score
                        conf = float(box.conf[0].cpu().numpy())
                        
                        # Get class name
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = "card"
                        
                        # Draw bounding box (green)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Draw label with confidence
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        
                        # Draw label background
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), (0, 255, 0), -1)
                        
                        # Draw label text
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        except Exception as e:
            print(f"Error during inference: {e}")
            continue
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = 30 / elapsed_time
            start_time = time.time()
            print(f"FPS: {fps:.1f}")
        
        # Display the frame
        cv2.imshow('Pokemon Card Detector - Press Q to Quit', frame)
        
        # Break loop on 'q' press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # Save current frame
            cv2.imwrite(f'card_detection_{int(time.time())}.jpg', frame)
            print("Frame saved!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released. Goodbye!")

if __name__ == "__main__":
    main() 