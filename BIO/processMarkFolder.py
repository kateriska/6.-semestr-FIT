import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob

import processSegmentation



# function for processing every neighbour pixel with LBP
def LBPprocesspixel(img, pix5, x, y):

    pixel_new_value = 0 # init the variable before try block

    try:
        if (img[x][y] >= pix5):
            pixel_new_value = 1
        else:
            pixel_new_value = 0

    except:
        pass

    return pixel_new_value

# function for processing LBP and gain new value for center pixel
def processLBP(img, x, y, lbp_values):
    # 3x3 window of pixels, where center pixel pix5 is on position [x,y]

    '''
            +------+------+------+
            | pix7 | pix8 | pix9 |
            +------+------+------+
    y-axis  | pix4 | pix5 | pix6 |
            +------+------+------+
            | pix1 | pix2 | pix3 |
            +------+------+------+
                    x-axis
    '''

    value_dec = 0 # init variable for computing the final new decimal value for center pixel

    pix5 = img[x][y] # center pixel on position [x,y]

    # process the all neighbour pixels and receive 8-bit binary code
    pix8 = LBPprocesspixel(img, pix5, x, y+1) # LSB
    pix7 = LBPprocesspixel(img, pix5, x-1, y+1)
    pix4 = LBPprocesspixel(img, pix5, x-1, y)
    pix1 = LBPprocesspixel(img, pix5, x-1, y-1)
    pix2 = LBPprocesspixel(img, pix5, x, y-1)
    pix3 = LBPprocesspixel(img, pix5, x+1, y-1)
    pix6 = LBPprocesspixel(img, pix5, x+1, y)
    pix9 = LBPprocesspixel(img, pix5, x+1, y+1) # MSB

    # compute new decimal value for center pixel - convert binary code to decimal number
    value_dec = (pix9 * 2 ** 7) + (pix6 * 2 ** 6) + (pix3 * 2 ** 5) + (pix2 * 2 ** 4) + (pix1 * 2 ** 3) + (pix4 * 2 ** 2) + (pix7 * 2 ** 1) + (pix8 * 2 ** 0)

    lbp_values.append(value_dec) # append new decimal value of pixel to array of whole processed lbp image
    return value_dec

# function for computing average gray level value of LBP for block of pixels
def computeAverageColorForBlock(lbp_image):
    rows = np.size(lbp_image, 0)
    cols = np.size(lbp_image, 1)

    block_div = 5
    step = 10
    white_block_pixels = [[]]

    for i in range(block_div, rows-block_div, step):
        for j in range(block_div, cols-block_div, step):
            sum_gray_level_pixels = 0.0
            pixel_count = 0
            for u in range(i-block_div, i+block_div):
                for v in range(j-block_div, j+block_div):
                    gray_level_value = lbp_image[u][v][0]
                    pixel_count += 1
                    sum_gray_level_pixels += gray_level_value

            average_block_color = round(sum_gray_level_pixels / pixel_count)

            for u in range(i-block_div, i+block_div):
                for v in range(j-block_div, j+block_div):
                    lbp_image[u, v] = [average_block_color,average_block_color,average_block_color]

    for i in range(0, rows):
        for j in range(0, cols):
            gray_level_value = lbp_image[i][j][0]
            if (204 <= gray_level_value <= 255):
                white_block_pixels.append([i,j]) # save white or very light gray blocks of pixels to list


    return white_block_pixels

# function for marking diseased blocks in fingerprint
def markWhiteBlocks(img, white_block_pixels, background_pixels):
    rows = np.size(img, 0)
    cols = np.size(img, 1)

    filtered_blocks = [[]] # init array for saving filtered pixels

    # filter pixels which are in white_block_pixels but not in background_pixels
    for p in white_block_pixels:
        if p not in background_pixels:
            print("Coordinates of processed diseased pixel: " + str(p))
            filtered_blocks.append(p)

    for i in range(0, rows):
        for j in range(0, cols):
            pixel_coordinate_list = []
            pixel_coordinate_list.append(i)
            pixel_coordinate_list.append(j)

            if pixel_coordinate_list in filtered_blocks:
                print("Marking pixel as diseased: " + str(pixel_coordinate_list))
                img[i, j] = [220,20,60]
    return img

def getMarkFolder(folder_id):

    # choose dataset of eczema (1) or verrucas (2) for processing
    if (folder_id == "1"):
        folder_path = './dataset/dataset1/*'
    elif (folder_id == "2"):
        folder_path = './dataset/dataset2/*'

    for file in glob.glob(folder_path):
        file_substr = file.split('/')[-1] # get name of processed file
        img = cv2.imread(file,0)
        img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
        background_pixels = processSegmentation.findBackgroundPixels(img)
        img = processSegmentation.imgSegmentation(img)
        cv2.imwrite("./processedImg/segImg.png", img)

        img = cv2.imread('./processedImg/segImg.png',0)
        #img = cv2.GaussianBlur(img,(5,5),0)

        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        cv2.imwrite("./processedImg/claheImg.png", img)

        img = cv2.imread('./processedImg/claheImg.png')
        height, width, channel = img.shape
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp_image = np.zeros((height, width,3), np.uint8)

        # processing LBP algorithm
        lbp_values = []
        for i in range(0, height):
            for j in range(0, width):
                lbp_image[i, j] = processLBP(img_gray, i, j, lbp_values)

        white_block_pixels = computeAverageColorForBlock(lbp_image)
        marked_img = markWhiteBlocks(img, white_block_pixels, background_pixels)
        # save marked images to folder
        cv2.imwrite("./markedImg/myMarkedImg/" + file_substr, marked_img)

    return
