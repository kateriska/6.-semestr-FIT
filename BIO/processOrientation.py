import numpy as np
import cv2
import processSegmentation
import math
from matplotlib import pyplot as plt

def computeAngle(phi_x, phi_y):
    angle = 0
    right_angle  = math.pi / 2

    if (phi_x > 0 and phi_y >=0):
        angle = math.atan(phi_y / phi_x)
    elif (phi_x > 0 and phi_y < 0):
        angle = math.atan(phi_y / phi_x) + 2 * np.pi
    elif (phi_x < 0):
        angle = math.atan(phi_y / phi_x) + np.pi
    elif (phi_x == 0 and phi_y > 0):
        angle = right_angle
    elif (phi_x == 0 and phi_y < 0):
        angle = 3 * right_angle

    return angle

def orientFieldEstimation(orig_img, height, width):
    white = cv2.imread("./processedImg/white.jpg")
    right_angle_pixels = []

    img = np.float32(orig_img)

    rows = np.size(img, 0)
    cols = np.size(img, 1)

    shape_img = img.shape

    white = cv2.resize(white,(rows, cols))

    grad_x = cv2.Sobel(img,cv2.CV_32F,1, 0, cv2.BORDER_DEFAULT, ksize=3)
    grad_y = cv2.Sobel(img,cv2.CV_32F,0, 1, cv2.BORDER_DEFAULT, ksize=3)
    print(grad_x)

    block_div = 7
    right_angle  = math.pi / 2
    step = 14

    orientation = np.zeros([int(rows / step), int(cols / step)], dtype = float)
    print(orientation.shape)
    color_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)


    for i in range(block_div, rows-block_div, step):
        for j in range(block_div, cols-block_div, step):
            sum_Vx = 0.0
            sum_Vy = 0.0
            for u in range(i-block_div, i+block_div):
                for v in range(j-block_div, j+block_div):
                    sum_Vx = sum_Vx + (2*grad_x[u][v] * grad_y[u][v])
                    sum_Vy = sum_Vy + ((grad_x[u][v] * grad_x[u][v]) - (grad_y[u][v] * grad_y[u][v]))
                    #print(sum_Vx)
            angle = computeAngle(sum_Vy, sum_Vx) / 2 + right_angle
            print(angle)
            if angle >= math.pi:
                angle -= math.pi
            orientation[int(i / step)][int(j / step)] = angle

            if (angle == math.pi / 2):
                for u in range(i-block_div, i+block_div):
                    for v in range(j-block_div, j+block_div):
                        right_angle_pixels.append([u,v])

            # draw to color image
            phi_x = math.cos(angle) * (block_div - 1)
            phi_y = math.sin(angle) * (block_div - 1)
            cv2.line(color_img, (int(j + phi_x), int(i + phi_y)), (int(j - phi_x), int(i - phi_y)), (0, 0, 255), 1)
            cv2.line(white, (int(j + phi_x), int(i + phi_y)), (int(j - phi_x), int(i - phi_y)), (0, 0, 255), 1)




    print(orientation)
    print(orientation.shape)
    #cv2.imshow('Orientation Field', white)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return color_img, right_angle_pixels

def getOrientationFeatures(file):
    img = cv2.imread(file,0)
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    img = processSegmentation.imgSegmentation(img)
    cv2.imwrite("./processedImg/segImg.png", img)

    img = cv2.imread('./processedImg/segImg.png',0)
    #img = cv2.GaussianBlur(img,(5,5),0)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    cv2.imwrite("./processedImg/claheImg.png", img)

    img = cv2.imread('./processedImg/claheImg.png',0)
    img_clahe = cv2.imread('./processedImg/claheImg.png',0)

    height = img.shape[0]
    width = img.shape[1]

    oriented_image, right_angle_pixels = orientFieldEstimation(img, height, width)
    print(right_angle_pixels)
    '''

    while True:
        cv2.imshow("Orientation Field", oriented_image)
        key = cv2.waitKey(1) & 0xFF
    # if the q key was pressed, break from the loop
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    '''


    #cv2.imshow('Oriented thinned image', oriented_image)
    '''
    figure = plt.figure(figsize=(30, 30))
    original_plot = figure.add_subplot(1,2,1)
    original_plot.set_title("Preprocessed Image", fontsize=10)
    original_plot.imshow(img_clahe, cmap='Greys',  interpolation='nearest')

    image_plot = figure.add_subplot(1,2,2)
    image_plot.set_title("Orientation Field", fontsize=10)
    image_plot.imshow(oriented_image)
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return right_angle_pixels
