import numpy as np
import cv2
import processSegmentation
import math
from matplotlib import pyplot as plt

def orientFieldEstimation(orig_img, height, width):
    white = cv2.imread("./processedImg/white.jpg")
    white = cv2.resize(white,(height, width))
    img = np.float32(orig_img)

    rows = np.size(img, 0)
    cols = np.size(img, 1)

    shape_img = img.shape

    grad_x = np.zeros(shape_img, dtype=np.float32)
    grad_y = np.zeros(shape_img, dtype=np.float32)
    Vx = np.zeros(shape_img, dtype=np.float32)
    Vy = np.zeros(shape_img, dtype=np.float32)
    theta = np.zeros(shape_img, dtype=np.float32)
    phi_x_array = np.zeros(shape_img, dtype=np.float32)
    phi_y_array = np.zeros(shape_img, dtype=np.float32)
    magnitude_array = np.zeros(shape_img, dtype=np.float32)

    grad_x = cv2.Sobel(img,cv2.CV_32FC1,1, 0, cv2.BORDER_DEFAULT, ksize=3)
    grad_y = cv2.Sobel(img,cv2.CV_32FC1,0, 1, cv2.BORDER_DEFAULT, ksize=3)

    block_div = 7
    right_angle  = math.pi / 2
    step = 14

    m = 0
    n = 0

    for i in range(block_div, rows-block_div, step):
        for j in range(block_div, cols-block_div, step):
            sum_Vx = 0.0
            sum_Vy = 0.0
            for u in range(i-block_div, i+block_div):
                for v in range(j-block_div, j+block_div):
                    #print(grad_x[u][v])
                    grad_x_value_np = grad_x[u][v]
                    grad_x_value_str = np.array2string(grad_x_value_np)
                    grad_x_value_str = grad_x_value_str.split("[", 1)[1]
                    grad_x_value_str = grad_x_value_str.split(".", 1)[0]
                    grad_x_value_str = grad_x_value_str.strip();
                    grad_x_value = int (grad_x_value_str)

                    grad_y_value_np = grad_y[u][v]
                    grad_y_value_str = np.array2string(grad_y_value_np)
                    grad_y_value_str = grad_y_value_str.split("[", 1)[1]
                    grad_y_value_str = grad_y_value_str.split(".", 1)[0]
                    grad_y_value_str = grad_y_value_str.strip();
                    grad_y_value = int (grad_y_value_str)


                    sum_Vx = sum_Vx + ((grad_x_value * grad_x_value) - (grad_y_value * grad_y_value))
                    sum_Vy = sum_Vy + ((2*grad_x_value * grad_y_value))

            if (sum_Vx != 0):
                #tan_arg = sum_Vy / sum_Vx
                result = 0.5 * cv2.fastAtan2(sum_Vy, sum_Vx);
                #print(result)
            #orientatin_matrix[i][j] = result
            else:
                result = 0.0

            magnitude = math.sqrt((sum_Vx * sum_Vx) + (sum_Vy * sum_Vy))
            phi_x = magnitude * math.cos(2*(math.radians(result)))
            phi_y = magnitude * math.sin(2*(math.radians(result)))
            if (phi_x != 0):
                orient = 0.5 * cv2.fastAtan2(phi_y, phi_x)
            else:
                orient = 0.0

            #print(orient)
            #orient_arr.append(orient)

            X0 = i + block_div
            Y0 = j + block_div
            r = block_div

            #result_rad = result * math.pi / 180.0
            orient_deg = orient - 90
            orient_rad = math.radians(orient_deg)

            X1 = r * math.cos(orient_rad)+ X0
            X1 = int (X1)
            #print(X1)

            Y1 = r * math.sin(orient_rad)+ Y0
            Y1 = int (Y1)

            X2 = X0 - r * math.cos(orient_rad)
            X2 = int (X2)

            Y2 = Y0 - r * math.sin(orient_rad)
            Y2 = int (Y2)

            orient_img = cv2.line(orig_img,(X0,Y0) , (X1,Y1), (0,255,0), 3)
            #cv2.imshow('Oriented image', orient_img)
            white_img = cv2.line(white,(X0,Y0) , (X1,Y1), (0,255,0), 3)
            #cv2.imshow('Oriented skeleton', white_img)
            rotated_img = cv2.rotate(white_img, cv2.ROTATE_90_CLOCKWISE)
            #cv2.imshow('Oriented rotated skeleton', rotated_img)
            flip_horizontal_img = cv2.flip(rotated_img, 1)

    #print(orient_arr)
    #print(or_array)
    #print(len(orient_arr))
    return flip_horizontal_img

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

    img = cv2.imread('./processedImg/claheImg.png')
    img_clahe = cv2.imread('./processedImg/claheImg.png',0)

    height = img.shape[0]
    width = img.shape[1]

    oriented_image = orientFieldEstimation(img, height, width)
    #cv2.imshow('Oriented thinned image', oriented_image)

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
    return
