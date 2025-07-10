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

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
CORS(app)

# Supabase configuration
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_ANON_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

# Load card database
def load_card_database():
    try:
        with open('../card_database.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Warning: card_database.pkl not found. Using empty database.")
        return {}

card_database = load_card_database()

def compute_phash(pil_img):
    '''Computes the perceptual hash of a given PIL image'''
    return str(imagehash.phash(pil_img))

def match_card(phash, card_database):
    '''Finds the closest match in the database using Hamming distance.'''
    min_dist = float('inf')
    best_match = None
    for name, db_hash in card_database.items():
        dist = imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(db_hash)
        if dist < min_dist:
            min_dist = dist
            best_match = name
    return best_match, min_dist

def getContours(img, imgContour, originalImg, areaMin=15000, width=500, height=700):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    card_found = False
    # Draw all contours in blue for visualization
    cv2.drawContours(imgContour, contours, -1, (255, 0, 0), 2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if area > areaMin and len(approx) == 4:
            card_found = True
            # Draw the detected card contour in magenta (as before)
            cv2.drawContours(imgContour, [cnt], -1, (255, 0, 255), 7)
            rectX, rectY, rectW, rectH = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (rectX, rectY), (rectX + rectW, rectY + rectH), (0, 255, 0), 5)
            points = []
            for point in approx:
                x, y = point[0]
                points.append([x, y])
            card = np.float32(points)
            x1, y1 = points[0]
            x2, y2 = points[1]
            x4, y4 = points[3]
            if np.sqrt((x1 - x2)**2 + (y1 - y2)**2) < np.sqrt((x1 - x4)**2 + (y1 - y4)**2):
                cardWarped = np.float32([[width, 0], [0, 0], [0, height], [width, height]])
            else:
                cardWarped = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
            matrix = cv2.getPerspectiveTransform(card, cardWarped)
            imgOutput = cv2.warpPerspective(originalImg, matrix, (width, height))
            return imgContour, imgOutput
    return imgContour, None

def process_image(image, card_database):
    img = image.copy()
    imgContour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (5, 5), 0)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGray, 100, 140)
    kernel = np.ones((5, 5), np.uint8)
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    imgContour, card_img = getContours(imgDil, imgContour, img)

    # Return both the contour image and the detected card image
    # The card_img will be None if no card is detected
    return imgContour, card_img

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
    '''Process webcam frame using card_matcher.py logic'''
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
        
        # Use the exact same processing logic as card_matcher.py
        imgContour, card_img = process_image(img, card_database)
        
        # Extract card information and add text overlay
        label = "No card detected"
        match_time = 0
        
        if card_img is not None:
            # Convert the detected card image to PIL for pHash
            pil_img = Image.fromarray(cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)).resize((300, 420))
            phash = compute_phash(pil_img)
            
            # Match card
            start_time = time.time()
            match, dist = match_card(phash, card_database)
            match_time = time.time() - start_time
            
            if match and dist < 20:  # Threshold for good match
                card_id = match.split("_")[-1] if "_" in match else match
                price_info, image_url = get_card_price(card_id)
                label = f"{match} (dist: {dist}, price: {price_info})"
                
                # Print match info like in card_matcher.py
                print(label)
                if match_time > 0.01:
                    print(f"Match time: {match_time:.4f}s")
                
                # Add text overlay to contour image
                cv2.putText(imgContour, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                # Convert the contour image to base64 for display
                _, buffer = cv2.imencode('.jpg', imgContour)
                contour_image = base64.b64encode(buffer).decode('utf-8')
                return {
                    "success": True,
                    "card_name": match,
                    "distance": dist,
                    "price_info": price_info,
                    "image_url": image_url,
                    "match_time": match_time,
                    "confidence": "High" if dist < 10 else "Medium" if dist < 15 else "Low",
                    "card_detected": True,
                    "contour_image": f"data:image/jpeg;base64,{contour_image}"
                }
            else:
                label = f"No match found (dist: {dist})"
                print(f"No good match found. Best match: {match}, distance: {dist}")
        
        # Add text overlay to contour image (for both no card and no match cases)
        cv2.putText(imgContour, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        # Convert the contour image to base64 for display
        _, buffer = cv2.imencode('.jpg', imgContour)
        contour_image = base64.b64encode(buffer).decode('utf-8')
        return {
            "success": True,
            "card_detected": False,
            "message": label,
            "contour_image": f"data:image/jpeg;base64,{contour_image}"
        }
            
    except Exception as e:
        print(f"Error in process_webcam_frame: {str(e)}")  # Debug log
        import traceback
        traceback.print_exc()
        return {"error": f"Error processing frame: {str(e)}"}

def process_uploaded_image(image_data):
    '''Process uploaded image to detect and match cards'''
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
        
        # Convert to PIL for pHash
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((300, 420))
        phash = compute_phash(pil_img)
        
        # Match card
        start_time = time.time()
        match, dist = match_card(phash, card_database)
        match_time = time.time() - start_time
        
        if match and dist < 20:  # Threshold for good match
            card_id = match.split("_")[-1] if "_" in match else match
            price_info, image_url = get_card_price(card_id)
            
            return {
                "success": True,
                "card_name": match,
                "distance": dist,
                "price_info": price_info,
                "image_url": image_url,
                "match_time": match_time,
                "confidence": "High" if dist < 10 else "Medium" if dist < 15 else "Low"
            }
        else:
            return {
                "success": False,
                "error": "No card match found",
                "distance": dist if dist != float('inf') else "N/A",
                "match_time": match_time
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 