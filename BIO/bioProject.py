import sys
import processORB
import processLBP
import processWavelet
import processSobel
import processHarris
import processOrientation
import processLBPLoop


if (sys.argv[1] == "orb"):
    file = sys.argv[2]
    processORB.getORBfeatures(file)
elif (sys.argv[1] == "lbp"):
    file = sys.argv[2]
    right_angle_pixels = processOrientation.getOrientationFeatures(file)
    processLBP.getLBPfeatures(file, right_angle_pixels)
elif (sys.argv[1] == "wavelet"):
    file = sys.argv[2]
    processWavelet.getWaveletFeatures(file)
elif (sys.argv[1] == "sobel"):
    file = sys.argv[2]
    processSobel.getSobelFeatures(file)
elif (sys.argv[1] == "harris"):
    file = sys.argv[2]
    processHarris.getHarrisFeatures(file)
elif (sys.argv[1] == "orient"):
    file = sys.argv[2]
    processOrientation.getOrientationFeatures(file)
elif (sys.argv[1] == "mark"):
    processLBPLoop.getLBPfeatures()
