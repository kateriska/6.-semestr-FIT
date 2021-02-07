import sys
import processORB
import processLBP
import processWavelet
import processSobel
import processMarkFile
import processMarkFolder

def printHelp():
    print("FINGERPRINT FEATURES AND MARKING DISEASED REGIONS:")
    print("Run with following arguments:")
    print("markfile <img_path> - Mark diseased regions in fingerprint image")
    print("markfolder <folder_id> - Mark diseased regions for folder of images")
    print("<folder_id> could be 1 or 2 (1 - Eczema, 2 - Verrucas)")
    print("orb <img_path> - Find and show ORB features of fingerprint")
    print("lbp <img_path> - Show LBP image and histogram")
    print("Extract GLCM features:")
    print("wavelet <img_path> - Show details of image processed with Wavelet transform")
    print("sobel <img_path> - Show image processed by Sobel and Laplacian operator")

if (sys.argv[1] == "orb"):
    file = sys.argv[2]
    processORB.getORBfeatures(file)
elif (sys.argv[1] == "markfile"):
    file = sys.argv[2]
    processMarkFile.getMarkFile(file)
elif (sys.argv[1] == "lbp"):
    file = sys.argv[2]
    processLBP.getLBPfeatures(file)
elif (sys.argv[1] == "wavelet"):
    file = sys.argv[2]
    processWavelet.getWaveletFeatures(file)
elif (sys.argv[1] == "sobel"):
    file = sys.argv[2]
    processSobel.getSobelFeatures(file)
elif (sys.argv[1] == "markfolder"):
    folder_id = sys.argv[2]
    processMarkFolder.getMarkFolder(folder_id)
elif (sys.argv[1] == "help"):
    printHelp()
else:
    sys.stderr.write("ERROR - Wrong arguments - see bioProject.py help\n")
    exit(1)
