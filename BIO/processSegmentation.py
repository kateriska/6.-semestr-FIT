import numpy as np
import cv2

# function for segmentation of fingerprint with using Otsu thresholding
def imgSegmentation(img):
    ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((11,11), np.uint8)
    opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel) # use morphological operations
    result = cv2.add(img, opening) # add mask with input image

    return result

# function for finding pixel coordinates which are in background
def findBackgroundPixels(img):
    ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((40,40), np.uint8)
    opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel) # use morphological operations

    rows = np.size(img, 0)
    cols = np.size(img, 1)

    background_pixels = [[]]


    for i in range(0, rows):
        for j in range(0, cols):
            gray_level_value = opening[i][j]
            if (gray_level_value == 255):
                background_pixels.append([i,j])

    return background_pixels
