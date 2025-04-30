import cv2
import numpy as np
from PIL import Image
import imagehash
import pickle
import os
import itertools

def load_card_database(database_path):
    with open(database_path, 'rb') as f:
        return pickle.load(f)

def preprocess_image(image_path, use_adaptive=False, enhance_contrast=False, dilate_edges=True):
    img = cv2.imread(image_path)
    # Add a white border to help close contours at the image edge
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255,255,255])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if use_adaptive:
        edged = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    else:
        # Stricter thresholds to avoid detecting the entire image
        edged = cv2.Canny(blur, 50, 250)
    if dilate_edges:
        kernel = np.ones((5,5), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=2)
    return img, edged

def line_intersection(line1, line2):
    # Each line is (x1, y1, x2, y2)
    xdiff = (line1[0] - line1[2], line2[0] - line2[2])
    ydiff = (line1[1] - line1[3], line2[1] - line2[3])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # Lines do not intersect

    d = (det([line1[0], line1[1]], [line1[2], line1[3]]),
         det([line2[0], line2[1]], [line2[2], line2[3]]))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [int(x), int(y)]

def find_card_contour(edged, debug_img=None, debug=False, area_threshold=60000, max_area_ratio=0.7):
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_h, img_w = edged.shape[:2]
    img_area = img_h * img_w
    if debug and debug_img is not None:
        debug_all = debug_img.copy()
        idx = 1
        for cnt in contours[:10]:
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # Check if contour touches the border
            touches_border = (
                np.any(cnt[:,0,0] <= 1) or np.any(cnt[:,0,0] >= img_w-2) or
                np.any(cnt[:,0,1] <= 1) or np.any(cnt[:,0,1] >= img_h-2)
            )
            print(f"Contour {idx}: area={area}, points={len(approx)}, touches_border={touches_border}")
            if area > area_threshold and area < max_area_ratio * img_area and not touches_border:
                cv2.drawContours(debug_all, [cnt], -1, (0, 0, 255), 2)
            idx += 1
        cv2.imwrite("debug_all_contours.png", debug_all)
        print("Saved all large contours as debug_all_contours.png")
    # Try to find a 4-point contour first
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Skip if area is out of bounds
        if area < area_threshold or area > max_area_ratio * img_area:
            continue
        # Skip contours touching the image border
        if (
            np.any(cnt[:,0,0] <= 1) or np.any(cnt[:,0,0] >= img_w-2) or
            np.any(cnt[:,0,1] <= 1) or np.any(cnt[:,0,1] >= img_h-2)
        ):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    # If not found, try convex hull of the largest contour
    if len(contours) > 0:
        hull = cv2.convexHull(contours[0])
        if debug and debug_img is not None:
            hull_img = debug_img.copy()
            cv2.drawContours(hull_img, [hull], -1, (255, 0, 0), 2)
            cv2.imwrite("debug_hull.png", hull_img)
            print("Saved convex hull as debug_hull.png")
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        if len(approx) >= 4:
            # Try to find the largest quadrilateral from hull points
            from itertools import combinations
            max_area = 0
            best_quad = None
            for quad in combinations(approx, 4):
                quad = np.array(quad)
                area = cv2.contourArea(quad)
                if area > max_area:
                    max_area = area
                    best_quad = quad
            if best_quad is not None:
                return best_quad.reshape(-1, 1, 2)
    # If still not found, use Hough Transform to estimate lines and corners
    if debug and debug_img is not None:
        hough_img = debug_img.copy()
        lines = cv2.HoughLinesP(edged, 1, np.pi/180, threshold=80, minLineLength=60, maxLineGap=20)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(hough_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.imwrite("debug_hough_lines.png", hough_img)
            print("Saved Hough lines as debug_hough_lines.png")
            # Find all intersections
            intersections = []
            for l1, l2 in itertools.combinations(lines, 2):
                pt = line_intersection(l1[0], l2[0])
                if pt is not None:
                    # Only keep intersections inside the image
                    if 0 <= pt[0] < debug_img.shape[1] and 0 <= pt[1] < debug_img.shape[0]:
                        intersections.append(pt)
            intersections = np.array(intersections)
            if len(intersections) >= 4:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=4, random_state=0).fit(intersections)
                corners = kmeans.cluster_centers_.astype(np.int32)
                if debug_img is not None:
                    for x, y in corners:
                        cv2.circle(hough_img, (x, y), 10, (0, 0, 255), -1)
                    cv2.imwrite("debug_hough_corners.png", hough_img)
                    print("Saved estimated corners as debug_hough_corners.png")
                return corners.reshape(-1, 1, 2)
    return None

def identify_card(image_path, database_path, debug=False, top_n=5, use_adaptive=False):
    card_database = load_card_database(database_path)
    img, edged = preprocess_image(
        image_path, 
        use_adaptive=use_adaptive, 
        enhance_contrast=True, 
        dilate_edges=True
    )
    if debug:
        cv2.imshow("Original", img)
        cv2.imshow("Edged", edged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Pass img to find_card_contour for debugging
    card_contour = find_card_contour(edged, debug_img=img, debug=debug)
    if card_contour is None:
        print("No card contour found.")
        return None

    # Debug: Draw the contour on the original image
    if debug:
        img_contour = img.copy()
        cv2.drawContours(img_contour, [card_contour], -1, (0, 255, 0), 3)
        cv2.imshow("Detected Contour", img_contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("debug_detected_contour.png", img_contour)
        print("Saved detected contour as debug_detected_contour.png")
        print("Contour points:", card_contour.reshape(-1, 2))

    warped = four_point_transform(img, card_contour)
    pil_img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)).resize((300, 420))
    if debug:
        cv2.imshow("Warped", cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pil_img.save("debug_warped.png")
        print("Saved warped image as debug_warped.png")

    phash = compute_phash(pil_img)
    print(f"Computed hash for input: {phash}")

    # Show top N closest matches
    distances = []
    for name, db_hash in card_database.items():
        dist = imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(db_hash)
        distances.append((name, db_hash, dist))
    distances.sort(key=lambda x: x[2])
    print(f"Top {top_n} closest matches:")
    for i in range(min(top_n, len(distances))):
        print(f"{i+1}: {distances[i][0]} | Hash: {distances[i][1]} | Distance: {distances[i][2]}")

    best_match, min_dist = distances[0][0], distances[0][2]
    print(f"Best match: {best_match} (distance: {min_dist})")
    return best_match

def four_point_transform(image, pts):
    '''
    Applies a perspective transform to get a top-down view of the card.
    '''
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def compute_phash(pil_img):
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

if __name__ == "__main__":
    image_path = "super_potion.png"  # Replace with your test image
    database_path = "card_database.pkl"
    identify_card(image_path, database_path, debug=True, top_n=5)