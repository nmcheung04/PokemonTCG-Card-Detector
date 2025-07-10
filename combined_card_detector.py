import cv2
import numpy as np
from ultralytics import YOLO
import time
import imagehash
from PIL import Image
import os
import pickle

def load_card_database(database_path):
    '''
    Loads the card database from a pickle file
    '''
    with open(database_path, 'rb') as f:
        return pickle.load(f)

def compute_phash(pil_img):
    '''
    Computes the perceptual hash of a given PIL image
    '''
    return str(imagehash.phash(pil_img))

def match_card(phash, card_database):
    '''
    Finds the closest match in the database using Hamming distance.
    '''
    min_dist = float('inf')
    best_match = None
    for name, db_hash in card_database.items():
        dist = imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(db_hash)
        if dist < min_dist:
            min_dist = dist
            best_match = name
    return best_match, min_dist

def phash_comparison(image_roi, card_database):
    """
    Perform pHash comparison with the card database using the same logic as card_matcher.py
    """
    try:
        # Convert ROI to PIL Image and resize like in card_matcher
        roi_pil = Image.fromarray(cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB)).resize((300, 420))
        phash = compute_phash(roi_pil)
        
        # Use the same matching logic as card_matcher
        best_match, best_distance = match_card(phash, card_database)
        
        return best_match, best_distance
    
    except Exception as e:
        print(f"Error in pHash comparison: {e}")
        return None, float('inf')

def edge_detection_on_roi(image_roi):
    """
    Perform edge detection on a region of interest (ROI)
    This is where you'd put your existing card_matcher edge detection logic
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area (remove very small ones)
    min_area = 100
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return edges, filtered_contours

def main():
    # Load the trained YOLOv8 model
    print("Loading YOLOv8 model...")
    model = YOLO('best.pt')
    print("Model loaded successfully!")
    
    # Load card database from pickle file (same as card_matcher.py)
    database_path = "card_database.pkl"
    try:
        card_database = load_card_database(database_path)
        print(f"Card database loaded successfully with {len(card_database)} cards!")
    except FileNotFoundError:
        print(f"Error: Card database file '{database_path}' not found!")
        print("Please ensure you have a card_database.pkl file in the current directory.")
        return
    except Exception as e:
        print(f"Error loading card database: {e}")
        return
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
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
    print("Press 'q' to quit, 's' to save frame")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("Warning: Could not read frame, retrying...")
            time.sleep(0.1)
            continue
        
        # Create a copy for card matching results
        match_frame = frame.copy()
        
        # Run YOLOv8 inference on the frame
        try:
            results = model(frame, conf=0.3)
            
            # Process YOLO results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get confidence score
                        conf = float(box.conf[0].cpu().numpy())
                        
                        # Extract ROI for card matching
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:  # Check if ROI is valid
                            try:
                                # Perform pHash comparison on ROI using the same logic as card_matcher
                                best_match, best_distance = phash_comparison(roi, card_database)
                                
                                # Draw ROI rectangle (blue)
                                cv2.rectangle(match_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                
                                # Add card matching results (same format as card_matcher)
                                if best_match and best_distance < 20:  # Threshold for good match
                                    match_label = f"Match: {best_match}"
                                    distance_label = f"Distance: {best_distance}"
                                    color = (0, 255, 0)  # Green for good match
                                else:
                                    match_label = "No match found"
                                    distance_label = f"Distance: {best_distance if best_distance != float('inf') else 'N/A'}"
                                    color = (0, 0, 255)  # Red for no match
                                
                                # Draw labels
                                cv2.putText(match_frame, match_label, (x1, y2 + 20), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                cv2.putText(match_frame, distance_label, (x1, y2 + 40), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
                                # Also perform edge detection for visualization
                                edges, contours = edge_detection_on_roi(roi)
                                
                                # Draw contours (yellow)
                                for contour in contours:
                                    # Adjust contour coordinates to full frame
                                    contour[:, :, 0] += x1
                                    contour[:, :, 1] += y1
                                    cv2.drawContours(match_frame, [contour], -1, (0, 255, 255), 1)
                                
                            except Exception as e:
                                print(f"Error in card matching: {e}")
                                continue
        
        except Exception as e:
            print(f"Error during YOLO inference: {e}")
            continue
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = 30 / elapsed_time
            start_time = time.time()
            print(f"FPS: {fps:.1f}")
        
        # Display only the card matching frame
        cv2.imshow('Card Matcher - pHash Detection', match_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'card_matching_{int(time.time())}.jpg', match_frame)
            print("Card matching frame saved!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released. Goodbye!")

if __name__ == "__main__":
    main() 