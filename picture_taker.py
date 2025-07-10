import cv2
import os
import re

save_dir = 'card_pictures'
os.makedirs(save_dir, exist_ok=True)

def get_next_image_number(folder):
    image_files = os.listdir(folder)
    numbers = []
    pattern = re.compile(r'^card_(\d+)\.png$')
    for filename in image_files:
        match = pattern.match(filename)
        if match:
            numbers.append(int(match.group(1)))

    return max(numbers) + 1 if numbers else 0

img_counter = get_next_image_number(save_dir)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Press 's' to save an image, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    cv2.imshow("Webcam - Press 's' to save", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Save the frame
        img_name = f"card_{img_counter}.png"
        img_path = os.path.join(save_dir, img_name)
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        img_counter += 1

    elif key == ord('q'):
        print("Quitting.")
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
