import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob

import processSegmentation



for file in glob.glob('./dataset/dataset2/*'):
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
    cv2.imwrite("./preprocessedDataset/verruca/" + file_substr, img)
