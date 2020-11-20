# Author: Katerina Fortova
# Bachelor's Thesis: Liveness Detection on Touchless Fingerprint Scanner
# Academic Year: 2019/20
# File: processedSegmentation.py - process segmentation of fingerprint with using Otsu, Gaussian or Mean thresholding

import numpy as np
import cv2

# function for segmentation of fingerprint with using Otsu thresholding
def imgSegmentation(img):
    ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((11,11), np.uint8)
    opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel) # use morphological operations
    #cv2.Canny(opening, 100, 200)
    '''
    while True:
        cv2.imshow("Orientation Field", opening)
        key = cv2.waitKey(1) & 0xFF
    # if the q key was pressed, break from the loop
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    #x, y = (opening == 0).nonzero()
    '''


    #vals = x, y

    #mask = img < opening
    #ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #sel = np.zeros_like(img)
    #sel[mask] = img[mask]
    #sel = cv2.add(img, opening, mask=img) # add mask with input image

    #print(vals)
    '''
    max_y = max(vals[1])
    #max_y_index = np.where(vals[1] == max_y)
    min_x = min(vals[0])

    max_x = max(vals[0])
    #max_x_index = np.where(vals[0] == max_x)
    min_y = min(vals[1])
    vals_list = []
    for x, y in zip(vals[0], vals[1]):
        print(x)
        print(y)
        vals_list.append([x,y])

    print(vals_list)
    print(max_x)
    #print(max_y_index)
    print(min_x)
    '''


    result = cv2.add(img, opening) # add mask with input image

    return result

def findBackgroundPixels(img):
    ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((40,40), np.uint8)
    opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel) # use morphological operations
    '''
    while True:
        cv2.imshow("Orientation Field", opening)
        key = cv2.waitKey(1) & 0xFF
    # if the q key was pressed, break from the loop
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    '''

    rows = np.size(img, 0)
    cols = np.size(img, 1)

    background_pixels = [[]]

    print(opening)

    for i in range(0, rows):
        for j in range(0, cols):
            gray_level_value = opening[i][j]
            print(gray_level_value)
            if (gray_level_value == 255):
                background_pixels.append([i,j])

    return background_pixels




# function for segmentation of fingerprint with using adaptive Gaussian thresholding
def adaptiveSegmentationGaussian(img):
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,3,4)
    kernel = np.ones((21,21), np.uint8)
    opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN,kernel) # use morphological operations
    #opening = cv.dilate(img,kernel,iterations = 200000)
    cv2.Canny(opening, 100, 200)
    result = cv2.add(img, opening) # add mask with input image
    return result

# function for segmentation of fingerprint with using adaptive Mean thresholding
def adaptiveSegmentationMean(img):
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,7)
    kernel = np.ones((24,24), np.uint8)
    opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN,kernel) # use morphological operations

    cv2.Canny(opening, 100, 200)
    result = cv2.add(img, opening) # add mask with input image
    return result
