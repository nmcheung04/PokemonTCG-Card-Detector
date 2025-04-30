import cv2
import numpy as np
import imagehash
import pickle
import time
from PIL import Image
import requests
import os 
from dotenv import load_dotenv

load_dotenv()

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
        
def getContours(img, imgContour, originalImg, areaMin=15000, width=500, height=700):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    card_found = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if area > areaMin and len(approx) == 4:
            card_found = True
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

def get_card_price(card_id):
    '''
    Fetches the price information for a given card ID from the Pokemon TCG API
    '''
    api_key = os.getenv("POKEMON_API_KEY")
    url = f"https://api.pokemontcg.io/v2/cards/{card_id}"
    headers = {"X-Api-Key": api_key}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            updatedAt = data['data']['tcgplayer']['updatedAt']
            price = data['data']['tcgplayer']['prices']['normal']['market']
            return f"{updatedAt}, ${price:.2f}"
        else:
            return "N/A"
    except Exception as e:
        print(f"Error fetching card price: {e}")
        return "N/A"

def process_image(image, card_database):
    img = image.copy()
    imgContour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (5, 5), 0)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGray, 100, 140)
    kernel = np.ones((5, 5), np.uint8)
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    imgContour, card_img = getContours(imgDil, imgContour, img)

    label = "No card detected"
    match_time = 0
    if card_img is not None:
        pil_img = Image.fromarray(cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)).resize((300, 420))
        phash = compute_phash(pil_img)
        # Timer for matching
        t1 = time.time()
        match, dist = match_card(phash, card_database)
        t2 = time.time()
        match_time = t2 - t1

        card_id = match.split("_")[-1]
        price_info = get_card_price(card_id)
        label = f"{match} (dist: {dist}, price: {price_info})"

    print(f"Match time: {match_time:.4f}s")
    return imgContour, label

def main():
    database_path = "card_database.pkl"
    card_database = load_card_database(database_path)
    print("Choose input mode:")
    print("1. Webcam")
    print("2. Image file")
    mode = input("Enter 1 or 2: ").strip()
    if mode == "1":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            imgContour, label = process_image(frame, card_database)
            cv2.putText(imgContour, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Pokemon Card Detector", imgContour)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    elif mode == "2":
        img_path = input("Enter image file path: ").strip()
        img = cv2.imread(img_path)
        if img is None:
            print("Could not load image:", img_path)
            return
        imgContour, label = process_image(img, card_database)
        cv2.putText(imgContour, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Pokemon Card Detector", imgContour)
        print("Press any key in the image window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Invalid input.")

if __name__ == "__main__":
    main()