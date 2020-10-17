import sys
import processORB
import processLBP
import processWavelet
import processSobel
import processHarris
import processOrientation

file = sys.argv[2]
if (sys.argv[1] == "orb"):
    processORB.getORBfeatures(file)
elif (sys.argv[1] == "lbp"):
    processLBP.getLBPfeatures(file)
elif (sys.argv[1] == "wavelet"):
    processWavelet.getWaveletFeatures(file)
elif (sys.argv[1] == "sobel"):
    processSobel.getSobelFeatures(file)
elif (sys.argv[1] == "harris"):
    processHarris.getHarrisFeatures(file)
elif (sys.argv[1] == "orient"):
    processOrientation.getOrientationFeatures(file)
