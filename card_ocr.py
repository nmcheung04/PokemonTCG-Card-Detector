import cv2
import pytesseract
from PIL import Image
import numpy as np

def extract_name_from_card(card_img):
    """
    Given a cropped card image (numpy array), extract the card name using OCR.
    Assumes the name is near the top of the card.
    """
    # Crop the top region where the name is likely to be (adjust as needed)
    h, w, _ = card_img.shape
    name_region = card_img[0:int(0.12*h), int(0.08*w):int(0.92*w)]  # top 12%, crop sides

    # Convert to grayscale and threshold for better OCR
    gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Convert to PIL Image for pytesseract
    pil_img = Image.fromarray(thresh)

    # Run OCR
    custom_config = r'--oem 3 --psm 7'  # Assume a single line of text
    text = pytesseract.image_to_string(pil_img, config=custom_config)

    # Clean up the text
    card_name = text.strip()
    return card_name

if __name__ == "__main__":
    # Example usage: read an image file and extract the name
    img_path = input("Enter path to cropped card image: ").strip()
    img = cv2.imread(img_path)
    if img is None:
        print("Could not load image:", img_path)
    else:
        name = extract_name_from_card(img)
        print("Detected card name:", name)