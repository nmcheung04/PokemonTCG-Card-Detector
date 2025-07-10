import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # or your preferred resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
    exit(1)

height, width, _ = frame.shape
print(f"Camera frame size: {width} x {height}")

# Define ROI width and height (adjust as needed)
roi_width = 300
roi_height = 150

# Calculate top-left corner of ROI to center it
x = (width - roi_width) // 2
y = (height - roi_height) // 2

roi = (x, y, roi_width, roi_height)
print(f"Centered ROI: {roi}")

# Then you can use roi in your OCR loop
