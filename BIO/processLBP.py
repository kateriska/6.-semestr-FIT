import numpy as np
import cv2
from matplotlib import pyplot as plt

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

def findWhiteRegions(height, width, lbp_image):
    print(lbp_image.shape)
    for i in range(0, height):
        for j in range(0, width):
            #print(lbp_image[i][j])
            gray_level_value = lbp_image[i][j][0]
            #print(gray_level_value)
            if (gray_level_value == 255):
                lbp_image[i, j] = [220,20,60]

    return lbp_image

def computeAverageColorForBlock(lbp_image):
    rows = np.size(lbp_image, 0)
    cols = np.size(lbp_image, 1)

    block_div = 7
    step = 14
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
            #print(average_block_color)

            for u in range(i-block_div, i+block_div):
                for v in range(j-block_div, j+block_div):
                    if (average_block_color == 255):
                        np.append(white_block_pixels, [u,v])
                    lbp_image[u, v] = [average_block_color,average_block_color,average_block_color]


    for i in range(0, rows):
        for j in range(0, cols):
            gray_level_value = lbp_image[i][j][0]
            print(gray_level_value)
            if (238 <= gray_level_value <= 255):
                white_block_pixels.append([i,j])

    #print(white_block_pixels)


    return white_block_pixels

def markWhiteBlocks(img, white_block_pixels, right_angle_pixels):
    print(white_block_pixels)
    rows = np.size(img, 0)
    cols = np.size(img, 1)

    # filter pixels which are only in white_block_pixels and not right_angle_pixels
    marked_pixels = []
    print("White block pixels length:" + str(len(white_block_pixels)))
    '''
    for i in range(len(white_block_pixels)):
        #print(white_block_pixels[i])
        if white_block_pixels[i] not in right_angle_pixels:
            marked_pixels.append(white_block_pixels[i])
    #print(marked_pixels)
    print("Marked pixels length:" + str(len(marked_pixels)))
    '''


    for i in range(0, rows):
        for j in range(0, cols):
            pixel_coordinate_list = []
            pixel_coordinate_list.append(i)
            pixel_coordinate_list.append(j)
            print(pixel_coordinate_list)

            if pixel_coordinate_list in white_block_pixels:
                img[i, j] = [220,20,60]
    return img

def getLBPfeatures(file, right_angle_pixels):
    img = cv2.imread(file,0)
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
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

    hist_lbp = cv2.calcHist([lbp_image], [0], None, [256], [0, 256])

    #lbp_image = findWhiteRegions(height, width, lbp_image)
    white_block_pixels = computeAverageColorForBlock(lbp_image)
    #print(img.shape)
    #print(lbp_image.shape)
    marked_img = markWhiteBlocks(img, white_block_pixels, right_angle_pixels)

    # show LBP image and LBP histogram
    figure = plt.figure(figsize=(30, 30))
    image_plot = figure.add_subplot(1,2,1)
    image_plot.imshow(marked_img)
    image_plot.set_xticks([])
    image_plot.set_yticks([])
    image_plot.set_title("LBP image", fontsize=10)
    current_plot = figure.add_subplot(1, 2, 2)
    current_plot.plot(hist_lbp, color = (0, 0, 0.2))

    current_plot.set_xlim([0,256])
    current_plot.set_ylim([0,10000])
    current_plot.set_title("LBP histogram", fontsize=10)
    current_plot.set_xlabel("Intensity")
    current_plot.set_ylabel("Count of pixels")
    ytick_list = [int(i) for i in current_plot.get_yticks()]
    current_plot.set_yticklabels(ytick_list,rotation = 90)
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return
