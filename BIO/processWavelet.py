import cv2
import numpy as np
import pywt
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
import processSegmentation

def getWaveletFeatures(file):
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

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to float32 for more resolution for use with pywt
    img = np.float32(img)
    img /= 255

    # gaining LH, HL and HH image
    coeffs2 = pywt.dwt2(img, "db2")
    LL, (LH, HL, HH) = coeffs2

    plt.imshow(LL, interpolation="nearest", cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('./processedImg/LLimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)

    plt.imshow(LH, interpolation="nearest", cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('./processedImg/LHimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)

    plt.imshow(HL, interpolation="nearest", cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('./processedImg/HLimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)

    plt.imshow(HH, interpolation="nearest", cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('./processedImg/HHimg.png', transparent = True, bbox_inches='tight', pad_inches = 0)

    print("====GLCM Features:====")
    print()
     # PROCESS LL IMAGE - Approximation
    print("APPROXIMATION:")
    LL_img = cv2.imread('./processedImg/LLimg.png',0)
    image = img_as_ubyte(LL_img)

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
    print("HORIZONTAL DETAIL:")
    LH_img = cv2.imread('./processedImg/LHimg.png',0)
    image = img_as_ubyte(LH_img)

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
    print("VERTICAL DETAIL:")
    HL_img = cv2.imread('./processedImg/HLimg.png',0)
    image = img_as_ubyte(HL_img)

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

    # PROCESS HH IMAGE - Diagonal detail
    print()
    print("DIAGONAL DETAIL:")
    HH_img = cv2.imread('./processedImg/HHimg.png',0)
    image = img_as_ubyte(HH_img)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
    inds = np.digitize(image, bins)

    max_value = inds.max() + 1
    matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)

    contrast4 = greycoprops(matrix_coocurrence, 'contrast')
    print("Contrast:")
    print(round(float(contrast4),6))

    homogeneity4 = greycoprops(matrix_coocurrence, 'homogeneity')
    print("Homogeneity:")
    print(round(float(homogeneity4),6))

    energy4 = greycoprops(matrix_coocurrence, 'energy')
    print("Energy:")
    print(round(float(energy4),6))

    correlation4 = greycoprops(matrix_coocurrence, 'correlation')
    print("Correlation:")
    print(round(float(correlation4),6))

    dissimilarity4 = greycoprops(matrix_coocurrence, 'dissimilarity')
    print("Dissimilarity:")
    print(round(float(dissimilarity4),6))

    ASM4 = greycoprops(matrix_coocurrence, 'ASM')
    print("ASM:")
    print(round(float(ASM4),6))




    titles = ['Approximation (LL)', ' Horizontal detail (LH)',
          'Vertical detail (HL)', 'Diagonal detail (HH)']
    coeffs2 = pywt.dwt2(img, "db2")
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(11, 11))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()
