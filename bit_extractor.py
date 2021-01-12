import cv2
import numpy as np
import sys
import imutils

def is_frame_marker_border(border, errorPercentage = 60):
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
    # print("{} {}".format(np.count_nonzero(bitCell), bitCell.size))
    bit =  0 if 2 * np.count_nonzero(bitCell) > bitCell.size / 2 else 1
    # print(bit)
    # cv2.imshow("bit cell", bitCell)
    # cv2.waitKey(0)
    return bit

def pre_process_image(image):
    out = image.copy()
    
    #has 3 channels
    if len(out.shape) == 3 and out.shape[-1] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    # _, thres = cv2.threshold(out, np.mean(out), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thres = cv2.threshold(out, 110, 255, cv2.THRESH_BINARY)


    return thres

def extract_bit(image):
    out = pre_process_image(image)

    # if not is_valid_border(out):
    #     return None

    borderSize = out.shape[0] // 18
    inner = out[borderSize:-borderSize, borderSize:-borderSize] if is_valid_border(out) else out
    bitSize = inner.shape[0] // 20
    # cv2.imshow("Inner", inner) 
    bit_ranges = [[], [], [], []]
    for i in range(0, 20):
        #get top border bit
        bitCell = inner[0:bitSize, i * bitSize: (i + 1) * bitSize]
        bit_ranges[0].append(get_bit(bitCell))
        #get bottom border bit
        bitCell = inner[-bitSize: -1, i * bitSize: (i + 1) * bitSize]
        bit_ranges[2].append(get_bit(bitCell))
        #get left border bit
        bitCell = inner[i * bitSize: (i + 1) * bitSize, 0:bitSize]
        # bitCell[:] = 200 if (i // 2) % 2 == 0 else 100
        bit_ranges[3].append(get_bit(bitCell))
        #get right border bit
        bitCell = inner[i * bitSize: (i + 1) * bitSize, -bitSize:-1]
        bit_ranges[1].append(get_bit(bitCell))
        # bitCell[:] = 200# if i % 2 == 0 else 100
    
    # print(bit_ranges)
    #chane direct for left and bottom
    bit_ranges[2].reverse()
    bit_ranges[3].reverse()

    # cv2.imshow('Inner', inner) 
    #remove unnessesary at head and tail of bit_range
    for i in range(4):
        bit_ranges[i] = bit_ranges[i][1:-2]
        bit_ranges[i] = [bit_ranges[i][j] for j in range(0, len(bit_ranges[i]), 2)]
    return bit_ranges

def get_id(bit_ranges):

    zero_id = np.array([292, 246, 177, 472])
    if bit_ranges is None:
        return -1, 0
    #convert to decimal
    id_matrix = np.array([int("".join(str(i) for i in bit), 2) for bit in bit_ranges], dtype=np.int32)
    
    for rotation in range(4):
        xor = id_matrix ^ np.roll(zero_id, rotation)
        # print(xor)
        values, count = np.unique(xor, return_counts=True)
        # print(value)
        if(values.shape == (1, )):
            return values[0], rotation
        else:
            for value, freq in zip(values, count):
                if freq >= 2:
                    error_bit = (sum([str(bin(element)).count("1") for element in xor.tolist()]))
                    if error_bit <= 6:
                        return value, rotation
    # for i in range(dictionary.shape[0]):
        # print(dictionary[i][:])
        # for rotation in range(4):
        #     # rotated = np.roll(dictionary[i][:], rotation)
        #     # print(rotated)
        #     cmp = id_matrix == rotated
        #     if cmp.all():
        #         return i, rotation

        # break

    return -1, 0
        

if __name__ == "__main__":
    # image = cv2.imread('../markers/Marker000.png')
    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE) 
    # print(len(image.shape))
     
    # _, marker = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # image = cv2.resize(image, (180, 180), interpolation=cv2.INTER_AREA)
    # image = imutils.rotate(image, 270)
    image = cv2.flip(image, 1)
    # cv2.imwrite(sys.argv[1], image)
    bits = extract_bit(image)
    print(bits)
    print(get_id(bits))

    cv2.imshow('marker', image)
    cv2.imshow('thres', pre_process_image(image))
    cv2.waitKey(0)

