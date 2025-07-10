import cv2
import pytesseract
from PIL import Image
import numpy as np

def extract_name_from_card(card_img):
    """
    Given a card image (numpy array), extract the card name using OCR.
    Assumes the name is near the top of the card.
    """
    h, w, _ = card_img.shape
    name_region = card_img[0:int(0.12*h), int(0.08*w):int(0.92*w)]  # top 12%, crop sides

    gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    pil_img = Image.fromarray(thresh)
    custom_config = r'--oem 3 --psm 7'
    text = pytesseract.image_to_string(pil_img, config=custom_config)
    card_name = text.strip()
    return card_name

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Press 's' to scan for card name, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        cv2.imshow("Webcam - Press 's' to OCR card name", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            card_name = extract_name_from_card(frame)
            print("Detected card name:", card_name)

        elif key == ord('q'):
            print("Quitting.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()