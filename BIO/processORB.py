import numpy as np
import cv2
from matplotlib import pyplot as plt
import processSegmentation

def getORBfeatures(file):
    img = cv2.imread(file,0)
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    img = processSegmentation.imgSegmentation(img)
    cv2.imwrite("./processedImg/segImg.png", img)

    img = cv2.imread('./processedImg/segImg.png',0)
    #img = cv2.GaussianBlur(img,(5,5),0)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    orb_result = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    figure = plt.figure(figsize=(30, 30))
    image_plot = figure.add_subplot(1,1,1)
    image_plot.imshow(orb_result), plt.show()

    return
