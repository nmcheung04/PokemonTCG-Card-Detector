from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from supabase import create_client, Client
import os
import cv2
import numpy as np
import imagehash
import pickle
import time
from PIL import Image
import requests
from dotenv import load_dotenv
import base64
import io
import uuid
from datetime import datetime
import json
from friends import bp as friends_bp
import easyocr
from ultralytics import YOLO
import difflib
import warnings

# Suppress multiprocessing resource tracker warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*leaked semaphore objects")

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
CORS(app)

# Supabase configuration
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_ANON_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

# Load new card database
def load_card_database_dict():
    try:
        with open('../card_database_dict.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Warning: card_database_dict.pkl not found. Using empty database.")
        return {}

# Initialize YOLO model and EasyOCR reader globally
print("Loading YOLOv8 model...")
yolo_model = YOLO('../best.pt')
print("Model loaded successfully!")

print("Initializing EasyOCR...")
ocr_reader = easyocr.Reader(['en'])
print("EasyOCR initialized!")

card_database = load_card_database_dict()
print(f"Loaded card database with {len(card_database)} names.")

# Frame processing optimization
frame_counter = 0
PROCESS_EVERY_N_FRAMES = 5  # Only process every 5th frame (increased from 3)
last_processed_time = 0
MIN_PROCESSING_INTERVAL = 1.0  # Minimum seconds between processing (increased from 0.5)
last_detection_results = None  # Cache results
CACHE_DURATION = 2.0  # Use cached results for 2 seconds

# Single camera optimizations
SINGLE_CAMERA_MODE = True  # Set to True if only using one camera
if SINGLE_CAMERA_MODE:
    PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame for single camera
    MIN_PROCESSING_INTERVAL = 0.8  # Slightly faster processing
    CACHE_DURATION = 1.5  # Shorter cache for more responsive updates

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

def compute_phash(image):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((300, 420))
    return str(imagehash.phash(pil_img))

def find_closest_name(name, name_list):
    matches = difflib.get_close_matches(name, name_list, n=1, cutoff=0.0)
    return matches[0] if matches else None

def match_phash(phash, hash_list):
    min_dist = float('inf')
    best_entry = None
    for entry in hash_list:
        if not isinstance(entry, tuple) or len(entry) < 2:
            continue
        db_hash = entry[1]
        dist = imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(db_hash)
        if dist < min_dist:
            min_dist = dist
            best_entry = entry
    return best_entry, min_dist

def detect_card_in_image(img):
    """New card detection using YOLO + EasyOCR + hash matching"""
    try:
        # Get ROI
        (rx1, ry1, rx2, ry2, roi_width, roi_height), roi = get_centered_card_roi(img, aspect_ratio=0.714, margin=0.05)
        
        # Run YOLO with higher confidence for faster processing
        yolo_results = yolo_model(roi, conf=0.6, verbose=False)  # Increased from 0.3
        
        detected_cards = []
        
        for result in yolo_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Crop the detected card
                    card_crop = roi[y1:y2, x1:x2]
                    if card_crop.size == 0:
                        continue
                    
                    # Only run OCR on high-confidence detections
                    detected_name = None
                    if conf > 0.7:  # Only OCR if YOLO is very confident
                        # OCR on top 20% of ROI
                        name_h = int(0.2 * roi_height)
                        name_roi = roi[0:name_h, :]
                        name_roi_pre = preprocess_for_ocr(name_roi)
                        
                        try:
                            ocr_results = ocr_reader.readtext(name_roi_pre)
                        except Exception as e:
                            print(f"OCR error: {e}")
                            ocr_results = []
                        
                        # Get best OCR result
                        if ocr_results:
                            best_ocr = max(ocr_results, key=lambda x: x[2])
                            detected_name = best_ocr[1].strip()
                            print(f"EasyOCR detected: '{detected_name}' (confidence: {best_ocr[2]:.2f})")
                        else:
                            print("EasyOCR: No text detected")
                    
                    # Compute hash and match
                    phash = compute_phash(card_crop)
                    print(f"Computed pHash: {phash}")
                    match_start_time = time.time()
                    
                    match_info = None
                    hash_dist = None
                    match_name = None
                    match_filename = None
                    
                    if detected_name and detected_name in card_database:
                        print(f"Found exact match for '{detected_name}' in database")
                        print(f"All entries for '{detected_name}':")
                        for entry in card_database[detected_name]:
                            print(f"  {entry}")
                        best_entry, hash_dist = match_phash(phash, card_database[detected_name])
                        if best_entry:
                            match_name = detected_name
                            match_filename = best_entry[0]
                            match_info = best_entry
                            print(f"Best match: {best_entry[0]} (distance: {hash_dist})")
                    else:
                        # Try closest name
                        closest_name = find_closest_name(detected_name, list(card_database.keys())) if detected_name else None
                        if closest_name:
                            print(f"Closest name to '{detected_name}': '{closest_name}'")
                            print(f"All entries for '{closest_name}':")
                            for entry in card_database[closest_name]:
                                print(f"  {entry}")
                            best_entry, hash_dist = match_phash(phash, card_database[closest_name])
                            if best_entry:
                                match_name = closest_name
                                match_filename = best_entry[0]
                                match_info = best_entry
                                print(f"Best match: {best_entry[0]} (distance: {hash_dist})")
                        else:
                            print(f"No close name found for '{detected_name}'")
                    
                    match_time = time.time() - match_start_time
                    print(f"Match time: {match_time:.4f}s")
                    
                    if match_name and hash_dist is not None:
                        detected_cards.append({
                            'name': match_name,
                            'filename': match_filename,
                            'distance': hash_dist,
                            'confidence': conf,
                            'match_time': match_time,
                            'detected_name': detected_name,
                            'bbox': (x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1)
                        })
                        print("---")
        
        return detected_cards
        
    except Exception as e:
        print(f"Error in detect_card_in_image: {e}")
        return []

def get_card_price(card_id):
    '''Fetches the price information and image URL for a given card ID from the Pokemon TCG API'''
    api_key = os.getenv("POKEMON_API_KEY")
    url = f"https://api.pokemontcg.io/v2/cards/{card_id}"
    headers = {"X-Api-Key": api_key}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            card_data = data['data']
            updated_at = card_data['tcgplayer']['updatedAt']
            price = card_data['tcgplayer']['prices']['normal']['market']
            image_url = card_data['images']['small']  # Get the small image URL
            return f"{updated_at}, ${price:.2f}", image_url
        else:
            return "N/A", None
    except Exception as e:
        print(f"Error fetching card price: {e}")
        return "N/A", None

def process_webcam_frame(frame_data):
    '''Process webcam frame using new YOLO+EasyOCR+hash matching'''
    global frame_counter, last_processed_time, last_detection_results
    
    try:
        print(f"Starting frame processing...")  # Debug log
        
        # Decode base64 image
        frame_data = frame_data.split(',')[1] if ',' in frame_data else frame_data
        frame_bytes = base64.b64decode(frame_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Failed to decode frame")  # Debug log
            return {"error": "Could not decode frame"}
        
        # Frame skipping and time-based throttling
        current_time = time.time()
        frame_counter += 1
        
        should_process = (frame_counter % PROCESS_EVERY_N_FRAMES == 0 and 
                         current_time - last_processed_time >= MIN_PROCESSING_INTERVAL)
        
        # Check if we can use cached results
        use_cached = (last_detection_results is not None and 
                     current_time - last_processed_time < CACHE_DURATION)
        
        # Create display image with overlays
        display_img = img.copy()
        
        # Draw ROI box
        (rx1, ry1, rx2, ry2, roi_width, roi_height), _ = get_centered_card_roi(img, aspect_ratio=0.714, margin=0.05)
        cv2.rectangle(display_img, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        line_y = ry1 + int(0.2 * roi_height)
        cv2.line(display_img, (rx1, line_y), (rx2, line_y), (255, 0, 0), 2)
        
        if should_process and not use_cached:
            last_processed_time = current_time
            # Detect cards using new method
            detected_cards = detect_card_in_image(img)
            last_detection_results = detected_cards
        elif use_cached:
            detected_cards = last_detection_results
        else:
            detected_cards = []
        
        if detected_cards:
            # Use the best match (lowest distance)
            best_card = min(detected_cards, key=lambda x: x['distance'])
            
            if best_card['distance'] < 50:  # Threshold for good match (increased from 20)
                # Draw bounding box
                fx1, fy1, fx2, fy2 = best_card['bbox']
                cv2.rectangle(display_img, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                
                # Get price info
                card_id = best_card['filename'].split("_")[-1].replace('.png', '') if best_card['filename'] else best_card['name']
                price_info, image_url = get_card_price(card_id)
                
                label = f"{best_card['name']} (dist: {best_card['distance']}, price: {price_info})"
                cv2.putText(display_img, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                # Convert display image to base64
                jpeg_quality = 60 if SINGLE_CAMERA_MODE else 80  # Lower quality for single camera
                _, buffer = cv2.imencode('.jpg', display_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                display_image = base64.b64encode(buffer).decode('utf-8')
                
                return {
                    "success": True,
                    "card_name": best_card['name'],
                    "filename": best_card['filename'],
                    "distance": best_card['distance'],
                    "price_info": price_info,
                    "image_url": image_url,
                    "match_time": best_card['match_time'],
                    "confidence": "High" if best_card['distance'] < 10 else "Medium" if best_card['distance'] < 15 else "Low",
                    "card_detected": True,
                    "contour_image": f"data:image/jpeg;base64,{display_image}"
                }
            else:
                label = f"No good match found (best dist: {best_card['distance']})"
                cv2.putText(display_img, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            if use_cached:
                label = "Using cached results..."
            elif should_process:
                label = "No card detected"
            else:
                label = "Processing..."
            cv2.putText(display_img, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        # Convert display image to base64 with reduced quality for better performance
        jpeg_quality = 60 if SINGLE_CAMERA_MODE else 80  # Lower quality for single camera
        _, buffer = cv2.imencode('.jpg', display_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        display_image = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "card_detected": False,
            "message": label,
            "contour_image": f"data:image/jpeg;base64,{display_image}"
        }
            
    except Exception as e:
        print(f"Error in process_webcam_frame: {str(e)}")  # Debug log
        import traceback
        traceback.print_exc()
        return {"error": f"Error processing frame: {str(e)}"}

def process_uploaded_image(image_data):
    '''Process uploaded image using new YOLO+EasyOCR+hash matching'''
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Could not decode image"}
        
        # Resize image for processing
        img = cv2.resize(img, (800, 600))
        
        # Detect cards using new method
        detected_cards = detect_card_in_image(img)
        
        if detected_cards:
            # Use the best match (lowest distance)
            best_card = min(detected_cards, key=lambda x: x['distance'])
            
            if best_card['distance'] < 50:  # Threshold for good match (increased from 20)
                # Get price info
                card_id = best_card['filename'].split("_")[-1].replace('.png', '') if best_card['filename'] else best_card['name']
                price_info, image_url = get_card_price(card_id)
                
                return {
                    "success": True,
                    "card_name": best_card['name'],
                    "filename": best_card['filename'],
                    "distance": best_card['distance'],
                    "price_info": price_info,
                    "image_url": image_url,
                    "match_time": best_card['match_time'],
                    "confidence": "High" if best_card['distance'] < 10 else "Medium" if best_card['distance'] < 15 else "Low"
                }
            else:
                return {
                    "success": False,
                    "error": "No good card match found",
                    "distance": best_card['distance'],
                    "match_time": best_card['match_time']
                }
        else:
            return {
                "success": False,
                "error": "No card detected",
                "distance": "N/A",
                "match_time": 0
            }
            
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        try:
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            # Store only serializable user data in session
            user_info = {
                "id": response.user.id,
                "email": response.user.email
            }
            session['user'] = user_info
            return jsonify({"success": True, "user": user_info})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        try:
            # Sign up with email confirmation disabled for easier testing
            response = supabase.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "email_confirm": False  # Disable email confirmation for now
                }
            })
            
            # Check if user was created successfully
            if response.user:
                # Auto-sign in the user after registration
                user_info = {
                    "id": response.user.id,
                    "email": response.user.email
                }
                session['user'] = user_info
                return jsonify({
                    "success": True, 
                    "message": "Account created successfully! You are now logged in.",
                    "user": user_info
                })
            else:
                return jsonify({"success": False, "error": "Registration failed - no user returned"})
                
        except Exception as e:
            print(f"Registration error: {str(e)}")  # Debug logging
            return jsonify({"success": False, "error": str(e)})
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user']['id']
    
    # Get user's deck
    try:
        response = supabase.table('user_decks').select('*').eq('user_id', user_id).execute()
        deck_cards = response.data
    except Exception as e:
        deck_cards = []
    
    return render_template('dashboard.html', deck_cards=deck_cards)

@app.route('/process-webcam-frame', methods=['POST'])
def process_webcam_frame_endpoint():
    if 'user' not in session:
        return jsonify({"success": False, "error": "Not authenticated"})
    
    data = request.get_json()
    frame_data = data.get('frame')
    
    # Process the webcam frame
    result = process_webcam_frame(frame_data)
    
    return jsonify(result)

@app.route('/upload-card', methods=['POST'])
def upload_card():
    if 'user' not in session:
        return jsonify({"success": False, "error": "Not authenticated"})
    
    data = request.get_json()
    image_data = data.get('image')
    
    # Process the image
    result = process_uploaded_image(image_data)
    
    if result.get('success'):
        # Save to user's deck
        user_id = session['user']['id']
        card_data = {
            'user_id': user_id,
            'card_name': result['card_name'],
            'card_id': result['card_name'].split("_")[-1] if "_" in result['card_name'] else result['card_name'],
            'price_info': result['price_info'],
            'distance': result['distance'],
            'confidence': result['confidence'],
            'added_at': datetime.now().isoformat(),
            'image_data': image_data  # Store the image data
        }
        
        try:
            supabase.table('user_decks').insert(card_data).execute()
            result['message'] = "Card added to your deck!"
        except Exception as e:
            result['error'] = f"Error saving to deck: {str(e)}"
    
    return jsonify(result)

@app.route('/api/deck')
def get_deck():
    if 'user' not in session:
        return jsonify({"success": False, "error": "Not authenticated"})
    
    user_id = session['user']['id']
    
    try:
        response = supabase.table('user_decks').select('*').eq('user_id', user_id).order('added_at', desc=True).execute()
        return jsonify({"success": True, "deck": response.data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/add-card', methods=['POST'])
def add_card():
    if 'user' not in session:
        return jsonify({"success": False, "error": "Not authenticated"})
    
    data = request.get_json()
    user_id = session['user']['id']
    
    card_data = {
        'user_id': user_id,
        'card_name': data.get('card_name'),
        'card_id': data.get('card_id'),
        'price_info': data.get('price_info'),
        'distance': data.get('distance'),
        'confidence': data.get('confidence'),
        'added_at': data.get('added_at'),
        'image_url': data.get('image_url')  # Add image URL field
    }
    
    try:
        supabase.table('user_decks').insert(card_data).execute()
        return jsonify({"success": True, "message": "Card added to deck"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/remove-card', methods=['POST'])
def remove_card():
    if 'user' not in session:
        return jsonify({"success": False, "error": "Not authenticated"})
    
    data = request.get_json()
    card_id = data.get('card_id')
    
    try:
        supabase.table('user_decks').delete().eq('id', card_id).execute()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/stats')
def get_stats():
    if 'user' not in session:
        return jsonify({"success": False, "error": "Not authenticated"})
    
    user_id = session['user']['id']
    
    try:
        response = supabase.table('user_decks').select('*').eq('user_id', user_id).execute()
        deck = response.data
        
        total_cards = len(deck)
        total_value = 0
        rarity_counts = {}
        
        for card in deck:
            price_str = card.get('price_info', 'N/A')
            if price_str != 'N/A' and '$' in price_str:
                try:
                    price = float(price_str.split('$')[1].split(',')[0])
                    total_value += price
                except:
                    pass
        
        return jsonify({
            "success": True,
            "stats": {
                "total_cards": total_cards,
                "estimated_value": f"${total_value:.2f}",
                "rarity_breakdown": rarity_counts
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

app.register_blueprint(friends_bp)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 