
# coding: utf-8

import numpy as np
from skimage import io
from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import minmax_scale
from scipy.stats import kurtosis, skew
from scipy import stats

def CalculateGLCM(img):
    # Calculate histogram of image
    
    # 1. Load the image and crop the image for 1 pixel.  
    # 2. Calculate the GLCM text features.  
    #     Energy Contrast Dissimilarity Homogeneity -> Can compute  
    #     Entropy SumAverage AutoCorrelation -> Need Coding  
    #     Skewness, Kurtosis -> Based on Histogram

    # np.histogram returns (hist, bin_edges)
    img_hist, bin_edges = np.histogram(img, bins=256, 
                                       range=(-0.5, 255.5), density=True)

    # Calculate Skewness, Kurtosis of Image histogram
    img_skewness = skew(img_hist)
    img_kurtosis = kurtosis(img_hist)
    
    # Calculate GLCM Texture Features
    directions = np.arange(0, np.pi*2, np.pi/4)
    GLCM = greycomatrix(img, [1], directions, normed=True)

    eng = greycoprops(GLCM, 'ASM')
    cont = greycoprops(GLCM, 'contrast')
    diss = greycoprops(GLCM, 'dissimilarity')
    homo = greycoprops(GLCM, 'homogeneity')

    GLCM_sq = np.squeeze(GLCM)
    ent = []
    for i in range(0,8):
        # Load the slice
        GLCM_slice = GLCM_sq[:,:,i]
        GLCM_norm = GLCM_slice / GLCM_slice.sum()
        # Normalize the slice
        GLCM_ent = GLCM_norm
        GLCM_ent[GLCM_ent == 0] = 1

        ent_mat = - GLCM_ent * np.log(GLCM_ent)
        ent_temp = ent_mat.sum()
        ent.append(ent_temp)

    # Calculate SumAverage of GLCM

    sumaverage=[]
    for i in range(0,8):
        # Load the slice
        GLCM_slice = GLCM_sq[:,:,i]
        GLCM_norm = GLCM_slice / GLCM_slice.sum()
        # Normalize the slice

        sumaverage_mat = np.zeros(256*2)
        for x in range(0,256):
            for y in range(0,256):
                j=x+y
                sumaverage_mat[i] += j * GLCM_norm[y, x]
        sumaverage.append(sumaverage_mat.sum())

    # Calculate Autocorrelation of GLCM

    autocorrelation= np.zeros(8)
    for i in range(0,8):
        # Load the slice
        GLCM_slice = GLCM_sq[:,:,i]

        # Normalize the slice
        GLCM_norm = GLCM_slice / GLCM_slice.sum()

        for x in range(0,256):
            for y in range(0,256):
                autocorrelation[i] += x * y * GLCM_norm[y, x]

    # Use Average value of 8 values.

    img_eng = np.mean(eng)
    img_cont = np.mean(cont)
    img_diss = np.mean(diss)
    img_homo = np.mean(homo)
    img_ent = np.mean(ent)
    img_sumav = np.mean(sumaverage)
    img_autocorr = np.mean(autocorrelation)
    
    # Calculate mean intensity value of image
    img_mean = np.mean(img)
    
    return [img_eng, img_cont, img_diss, img_homo, img_ent, img_sumav, img_autocorr, img_skewness, img_kurtosis, img_mean]

