import numpy as np
import cv2
import processSegmentation

def getHarrisFeatures(file):
    img = cv2.imread(file,0)
    results = list()

    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    img = processSegmentation.imgSegmentation(img)
    cv2.imwrite("./processedImg/segImg.png", img)

    img = cv2.imread('./processedImg/segImg.png',0)
    #img = cv2.GaussianBlur(img,(5,5),0)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    cv2.imwrite("./processedImg/claheImg.png", img)

    img = cv2.imread('./processedImg/claheImg.png')

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    return
