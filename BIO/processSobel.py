import cv2
import numpy as np
from matplotlib import pyplot as plt
import processSegmentation
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte

def getSobelFeatures(file):
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(img,cv2.CV_64F) # get result of image processing with Laplacian operator
    cv2.imwrite("./processedImg/laplacian_img.png", laplacian)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # get result of x-axis for image processed with Sobel operator
    cv2.imwrite("./processedImg/sobelx_img.png", sobelx)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # get result of y-axis for image processed with Sobel operator
    cv2.imwrite("./processedImg/sobely_img.png", sobelx)

    print("====GLCM Features:====")
    print()
     # PROCESS LL IMAGE - Approximation
    print("LAPLACIAN:")
    laplacian_img = cv2.imread('./processedImg/laplacian_img.png',0)
    results.append(laplacian_img)
    image = img_as_ubyte(laplacian_img)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
    inds = np.digitize(image, bins)

    max_value = inds.max() + 1
    matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)

    # GLCM matrix characteristics - contrast, homogeinity, energy, correlation
    contrast = greycoprops(matrix_coocurrence, 'contrast')
    print("Contrast:")
    print(round(float(contrast),6))

    homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
    print("Homogeneity:")
    print(round(float(homogeneity),6))

    energy = greycoprops(matrix_coocurrence, 'energy')
    print("Energy:")
    print(round(float(energy),6))

    correlation = greycoprops(matrix_coocurrence, 'correlation')
    print("Correlation:")
    print(round(float(correlation),6))

    dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')
    print("Dissimilarity:")
    print(round(float(dissimilarity),6))

    ASM = greycoprops(matrix_coocurrence, 'ASM')
    print("ASM:")
    print(round(float(ASM),6))


    print()


     # PROCESS LH IMAGE - Horizontal detail
    print("SOBEL X:")
    sobelx_img = cv2.imread('./processedImg/sobelx_img.png',0)
    results.append(sobelx_img)
    image = img_as_ubyte(sobelx_img)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
    inds = np.digitize(image, bins)

    max_value = inds.max() + 1
    matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)

    # GLCM matrix characteristics - contrast, homogeinity, energy, correlation
    contrast2 = greycoprops(matrix_coocurrence, 'contrast')
    print("Contrast:")
    print(round(float(contrast2),6))

    homogeneity2 = greycoprops(matrix_coocurrence, 'homogeneity')
    print("Homogeneity:")
    print(round(float(homogeneity2),6))

    energy2 = greycoprops(matrix_coocurrence, 'energy')
    print("Energy:")
    print(round(float(energy2),6))

    correlation2 = greycoprops(matrix_coocurrence, 'correlation')
    print("Correlation:")
    print(round(float(correlation2),6))

    dissimilarity2 = greycoprops(matrix_coocurrence, 'dissimilarity')
    print("Dissimilarity:")
    print(round(float(dissimilarity2),6))

    ASM2 = greycoprops(matrix_coocurrence, 'ASM')
    print("ASM:")
    print(round(float(ASM2),6))

    print()


    # PROCESS HL IMAGE - Vertical detail
    print("SOBEL Y:")
    sobely_img = cv2.imread('./processedImg/sobely_img.png',0)
    results.append(sobely_img)
    image = img_as_ubyte(sobely_img)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
    inds = np.digitize(image, bins)

    max_value = inds.max() + 1
    matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)

    contrast3 = greycoprops(matrix_coocurrence, 'contrast')
    print("Contrast:")
    print(round(float(contrast3),6))

    homogeneity3 = greycoprops(matrix_coocurrence, 'homogeneity')
    print("Homogeneity:")
    print(round(float(homogeneity3),6))

    energy3 = greycoprops(matrix_coocurrence, 'energy')
    print("Energy:")
    print(round(float(energy3),6))

    correlation3 = greycoprops(matrix_coocurrence, 'correlation')
    print("Correlation:")
    print(round(float(correlation3),6))

    dissimilarity3 = greycoprops(matrix_coocurrence, 'dissimilarity')
    print("Dissimilarity:")
    print(round(float(dissimilarity3),6))

    ASM3 = greycoprops(matrix_coocurrence, 'ASM')
    print("ASM:")
    print(round(float(ASM3),6))


    titles = ['Applied Laplacian operator',
          'Applied Sobel operator of x axis', 'Applied Sobel operator of y axis']


    fig = plt.figure(figsize=(11, 11))
    i = 0
    for a in results:
        ax = fig.add_subplot(2, 2, i + 1)
        ax.imshow(a, cmap='gray')
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        i = i + 1

    fig.tight_layout()
    plt.show()
