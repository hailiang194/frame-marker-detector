import cv2
import numpy as np

THRESH_WINDOWS_SIZE = list(range(3, 24, 10))

distance = lambda x, y: np.linalg.norm(x - y)

def is_convex_contours(contour):

    return False

def filter_contours(image, contours):
    filtered = []
    for contour in contours:
        if contour.size < 0.03 * max(image.shape) or contour.size > 4.0 * max(image.shape):
            continue
        appox = cv2.approxPolyDP(contour, 0.05 * contour.size, True)
        if len(appox) != 4:
            continue

        
        filtered.append(appox)
    return filtered

def get_markers(image):
    
    image_clone = image.copy()
    if len(image.shape) == 3:
        image_clone = cv2.cvtColor(image_clone, cv2.COLOR_BGR2GRAY)
    all_contours = []
    for thres_size in THRESH_WINDOWS_SIZE:
        thres_img = cv2.adaptiveThreshold(image_clone, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, thres_size, 7)
        # lines = cv2.HoughLinesP(thres_img, 5, np.pi / 180, 150, minLineLength=100, maxLineGap=0)
        contours, hierachy = cv2.findContours(thres_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(filter_contours(image_clone, contours))
        # yield thres_img, filter_contours(image_clone, contours)
    return all_contours
