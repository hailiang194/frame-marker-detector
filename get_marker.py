import cv2
import numpy as np
import bit_extractor
from multiprocessing import Pool
import time

THRESH_WINDOWS_SIZE = list(range(3, 24, 10))


def filter_contours(image, contours):
    filtered = []
    for contour in contours:
        per = cv2.arcLength(contour, True)
        if per < 0.03 * max(image.shape) or per > 4.0 * max(image.shape):
            continue
        appox = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
        if len(appox) != 4 or not cv2.isContourConvex(appox):
            continue
        
        # print(appox[3][0]) 
        filtered.append(appox)
    return filtered

def rotate_contour(contour, rotation):
    for _ in range(rotation):
        contour_clone = contour.copy()
        contour[0][0], contour[1][0], contour[2][0], contour[3][0] = contour_clone[1][0], contour_clone[2][0], contour_clone[3][0], contour_clone[0][0]

        return contour

def is_duplicated_marker(marker1, marker2, min_distance_rate):
    firstM = cv2.moments(marker1)
    firstC = np.array([firstM["m10"] / firstM["m00"], firstM["m01"] / firstM["m00"]])
    secondM = cv2.moments(marker2)
    secondC = np.array([secondM["m10"] / secondM["m00"], secondM["m01"] / secondM["m00"]])

    return np.linalg.norm(firstC - secondC) <= min_distance_rate * cv2.arcLength(marker1, True)

def remove_duplicated_markers(ids, markers, min_distance_rate = 0.05):
    marker_dict = {}

    for marker_id, marker in zip(ids, markers):
        if marker_id not in marker_dict.keys():
            marker_dict[marker_id] = [marker]
        else:
            marker_dict[marker_id].append(marker)
   
    ids = []
    markers = []

    for marker_id in marker_dict.keys():
        unique_marker = [marker_dict[marker_id][0]]
        # print(marker_dict[marker_id][0])
        for i in range(len(marker_dict[marker_id])):
            if not any([is_duplicated_marker(marker_dict[marker_id][i], u_marker, min_distance_rate) for u_marker in unique_marker]):
                unique_marker.append(marker_dict[marker_id][i])

        for u_marker in unique_marker:
            ids.append(marker_id)
            markers.append(u_marker)
    

    return ids, markers

    # print(marker_dict)

def process_thresh(param):

    thres_img = cv2.adaptiveThreshold(param[0], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, param[1], 7)

    contours, hierachy = cv2.findContours(thres_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return filter_contours(param[0], contours)
    
def get_markers(image):
    
    # times = []
    # times.append(time.process_time())
    image_clone = image.copy()
    if len(image.shape) == 3:
        image_clone = cv2.cvtColor(image_clone, cv2.COLOR_BGR2GRAY)
    all_contours = []
    ids = []
    corners = []
    rejected = []
    # times.append(time.process_time())

    with Pool(1, maxtasksperchild=len(THRESH_WINDOWS_SIZE)) as pool:
        thresh_result = (pool.map(process_thresh, [[image_clone, window_size] for window_size in THRESH_WINDOWS_SIZE]))
        for result in thresh_result:
            all_contours.extend(result)

    # for thres_size in THRESH_WINDOWS_SIZE:
        # times.append(time.process_time())
        # thres_img = cv2.adaptiveThreshold(image_clone, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, thres_size, 7)
        # times.append(time.process_time())
        # lines = cv2.HoughLinesP(thres_img, 5, np.pi / 180, 150, minLineLength=100, maxLineGap=0)
        # contours, hierachy = cv2.findContours(thres_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # times.append(time.process_time())
        # all_contours.extend(filter_contours(image_clone, contours))
        # times.append(time.process_time())
        # yield thres_img, filter_contours(image_clone, contours)
        # print([times[i] - times[i - 1] for i in range(1, len(times))])
        # times = []

    # times.append(time.process_time())
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
            warp = cv2.warpPerspective(image_clone, M, (maxWidth, maxHeight))
            bits = bit_extractor.extract_bit(warp)
            marker_id, rotation = bit_extractor.get_id(bits)
        #try to detect from transpose image
            if marker_id == -1:
                contour_clone = contour.copy()
                bits = bit_extractor.extract_bit(cv2.flip(warp, 1))
                contour[0][0], contour[1][0] = contour_clone[1][0], contour_clone[0][0]
                contour[2][0], contour[3][0] = contour_clone[3][0], contour_clone[2][0]
                marker_id, rotation = bit_extractor.get_id(bits)

            if marker_id != -1:
                for _ in range(rotation):
                    contour_clone = contour.copy()
                    contour[0][0], contour[1][0], contour[2][0], contour[3][0] = contour_clone[1][0], contour_clone[2][0], contour_clone[3][0], contour_clone[0][0]

                corners.append(contour)

            # corners.append(np.roll(cnt_matrix, -rotation).astype(int))
                ids.append(marker_id)
            else:
                rejected.append(contour)
    # times.append(time.process_time())
        ids, corners = remove_duplicated_markers(ids, corners)
    # times.append(time.process_time())
    # print(times)
    # print([times[i] - times[i - 1] for i in range(1, len(times))])
        return ids, corners, rejected
