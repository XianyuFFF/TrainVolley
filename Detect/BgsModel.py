import cv2
import os
import libbgs


class BgsModel:
    def __init__(self):
        ## BGS Library algorithms
        # bgs = libbgs.FrameDifference()
        # bgs = libbgs.StaticFrameDifference()
        # bgs = libbgs.AdaptiveBackgroundLearning()
        # bgs = libbgs.AdaptiveSelectiveBackgroundLearning()
        # bgs = libbgs.DPAdaptiveMedian()
        # bgs = libbgs.DPEigenbackground()
        # bgs = libbgs.DPGrimsonGMM()
        # bgs = libbgs.DPMean()
        # bgs = libbgs.DPPratiMediod()
        # bgs = libbgs.DPTexture()
        # bgs = libbgs.DPWrenGA()
        # bgs = libbgs.DPZivkovicAGMM()
        # bgs = libbgs.FuzzyChoquetIntegral()
        # bgs = libbgs.FuzzySugenoIntegral()
        # bgs = libbgs.GMG() # if opencv 2.x
        # bgs = libbgs.IndependentMultimodal()
        # bgs = libbgs.KDE()
        # bgs = libbgs.KNN() # if opencv 3.x
        # bgs = libbgs.LBAdaptiveSOM()
        # bgs = libbgs.LBFuzzyAdaptiveSOM()
        # bgs = libbgs.LBFuzzyGaussian()
        # bgs = libbgs.LBMixtureOfGaussians()
        # bgs = libbgs.LBSimpleGaussian()
        # bgs = libbgs.LBP_MRF()
        # bgs = libbgs.LOBSTER()
        # bgs = libbgs.MixtureOfGaussianV1() # if opencv 2.x
        # bgs = libbgs.MixtureOfGaussianV2()
        # bgs = libbgs.MultiCue()
        # bgs = libbgs.MultiLayer()
        # bgs = libbgs.PAWCS()
        # bgs = libbgs.PixelBasedAdaptiveSegmenter()
        # bgs = libbgs.SigmaDelta()
        # bgs = libbgs.SuBSENSE()
        # bgs = libbgs.T2FGMM_UM()
        # bgs = libbgs.T2FGMM_UV()
        # bgs = libbgs.T2FMRF_UM()
        # bgs = libbgs.T2FMRF_UV()
        # bgs = libbgs.VuMeter()
        # bgs = libbgs.WeightedMovingMean()
        # bgs = libbgs.WeightedMovingVariance()
        # bgs = libbgs.TwoPoints()
        # bgs = libbgs.ViBe()
        # bgs = libbgs.CodeBook()
        # bgs = cv2.createBackgroundSubtractorMOG2()

        self.bgs = cv2.createBackgroundSubtractorMOG2(varThreshold=48)
        self.current_time = 0

    def __call__(self, image):
        return self.bgs.apply(image)

