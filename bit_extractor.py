import cv2
import numpy as np
import sys

def is_frame_marker_border(border, errorPercentage = 33):
    return np.count_nonzero(border) < int(border.size * errorPercentage / 100.0)

def is_valid_border(image):
    borderHeigh = image.shape[0] // 18
    borderWidth = image.shape[1]
    
    #check head border
    border = image[0:borderHeigh, 0: borderWidth]
    if not is_frame_marker_border(border):
        return False
    #check tail border
    border = image[-borderHeigh : -1, -borderWidth: -1]
    # border[:] = 200
    if not is_frame_marker_border(border):
        return False
    ##check left border
    border = image[0:borderWidth, 0: borderHeigh]
    # border[:] = 100
    if not is_frame_marker_border(border):
        return False
    ##check right border
    border = image[-borderWidth: -1, -borderHeigh: -1]
    # border[:] = 100
    if not is_frame_marker_border(border):
        return False

    return True

def get_bit(bitCell):
    return 0 if np.count_nonzero(bitCell) > bitCell.size / 2 else 1

def extract_bit(image):
    if not is_valid_border(image):
        return None
    
    borderSize = image.shape[0] // 18
    inner = image[borderSize:-borderSize, borderSize:-borderSize]
    
    bitSize = inner.shape[0] // 20
    
    bit_ranges = [[], [], [], []]
    for i in range(20):
        #get top border bit
        bitCell = inner[0:bitSize, i * bitSize: (i + 1) * bitSize]
        bit_ranges[0].append(get_bit(bitCell))
        #get bottom border bit
        bitCell = inner[-bitSize: -1, i * bitSize: (i + 1) * bitSize]
        bit_ranges[1].append(get_bit(bitCell))
        #get left border bit
        bitCell = inner[i * bitSize: (i + 1) * bitSize, 0:bitSize]
        bit_ranges[2].append(get_bit(bitCell))
        #get right border bit
        bitCell = inner[i * bitSize: (i + 1) * bitSize, -bitSize:-1]
        bit_ranges[3].append(get_bit(bitCell))
        # bitCell[:] = 200 if i % 2 == 0 else 100
    
    #chane direct for left and bottom
    bit_ranges[1].reverse()
    bit_ranges[2].reverse()

    #remove unnessesary at head and tail of bit_range
    for i in range(4):
        bit_ranges[i] = bit_ranges[i][1:-2]
        bit_ranges[i] = [bit_ranges[i][j] for j in range(0, len(bit_ranges[i]), 2)]
    return bit_ranges


if __name__ == "__main__":
    # image = cv2.imread('../markers/Marker000.png')
    image = cv2.imread('/home/hailiang194/Downloads/frame-marker0.png', cv2.IMREAD_GRAYSCALE) 
     
    
    _, marker = cv2.threshold(image[34:34 + 243, 43:43 + 243], 0, 255, cv2.THRESH_BINARY)

    marker = cv2.resize(marker, (180, 180), interpolation=cv2.INTER_AREA)
    
    print(extract_bit(marker))

    cv2.imshow('marker', marker)
    cv2.waitKey(0)

