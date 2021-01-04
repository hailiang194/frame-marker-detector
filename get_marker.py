import cv2
import numpy as np
import bit_extractor


THRESH_WINDOWS_SIZE = list(range(3, 24, 10))

distance = lambda x, y: np.linalg.norm(x - y)

def filter_contours(image, contours):
    filtered = []
    for contour in contours:
        if contour.size < 0.03 * max(image.shape) or contour.size > 4.0 * max(image.shape):
            continue
        appox = cv2.approxPolyDP(contour, 0.05 * contour.size, True)
        if len(appox) != 4 or not cv2.isContourConvex(appox):
            continue
        
        if appox[1][0][0] > appox[2][0][0]:
            appox[1][0][:], appox[2][0][:] = appox[2][0][:], appox[1][0][:]
        # print(appox[3][0]) 
        filtered.append(appox)
    return filtered



def get_markers(image):
    
    image_clone = image.copy()
    if len(image.shape) == 3:
        image_clone = cv2.cvtColor(image_clone, cv2.COLOR_BGR2GRAY)
    all_contours = []
    ids = []
    corners = []
    rejected = []
    for thres_size in THRESH_WINDOWS_SIZE:
        thres_img = cv2.adaptiveThreshold(image_clone, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, thres_size, 7)
        # lines = cv2.HoughLinesP(thres_img, 5, np.pi / 180, 150, minLineLength=100, maxLineGap=0)
        contours, hierachy = cv2.findContours(thres_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(filter_contours(image_clone, contours))
        # yield thres_img, filter_contours(image_clone, contours)
    for contour in all_contours:
        maxWidth, maxHeight = (180, 180)
        dst = np.array([
	        [0, 0],
	        [maxWidth - 1, 0],
	        [maxWidth - 1, maxHeight - 1],
	        [0, maxHeight - 1]], dtype = "float32")
        cnt_matrix = np.array([contour[i][0] for i in range(4)], dtype=np.float32)
        # calculate the perspective transform matrix and warp
        # the perspective to grab the screen
        M = cv2.getPerspectiveTransform(cnt_matrix, dst)
        warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        bits = bit_extractor.extract_bit(warp)
        marker_id, rotation = bit_extractor.get_id(bits)
        if marker_id != -1:
            corners.append(np.roll(contour, rotation))
            ids.append(marker_id)
        else:
            rejected.append(contour)

    return ids, corners, rejected
