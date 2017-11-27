# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:21:25 2017

@author: Alex
"""

import cv2
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

def Correction(subsampled_img_K, network_output, substep=4, lowfreqPercent=0.04):
    """
    Corrects network output using the input subsampled image.

    Parameters
    ------------
    subsampled_img_K : numpy.core.memmap.memmap
        K-space subsampled image used to generate network input
    network_output: numpy.core.memmap.memmap
        Output from the CNN
    substep: int
        every substep-th line will be included (4 in paper)
    lowfrewPercent :  float
        percent of low frequencies to add into model (0.04 in paper)

    Returns
    ------------
    numpy.core.memmap.memmap
    An numpy image object representing a list of subsampled human-
    interpretable images.
    """

        # 2-dimensional fast Fourier transform
    t_output = np.fft.fft2(network_output)#_slice)

        # shifts 0 frequency to center
    tshift_output = np.fft.fftshift(t_output)
    tshift_input = subsampled_img_K
        #Subsampler,
        #accounts for the double-counted lines
    lowfreqModifiedPercent = 1.0/float(substep)*lowfreqPercent+lowfreqPercent

    start = len(tshift_output)/2-int(lowfreqModifiedPercent*float(len(tshift_output)))
    end = len(tshift_output)/2+int(lowfreqModifiedPercent*float(len(tshift_output)))
    print("starting")
    print([len(tshift_output), start, end])
    #print(tshift_input[0:10,0:20])
    #print(tshift_output[0:10,0:20])
    for i in range(0, start):
        if i % substep == 0:
            tshift_output[i] = tshift_input[i]
    for i in range (start, end):
        tshift_output[i] = tshift_input[i]
    for i in range (end, len(tshift_output)):
        if i % substep == 0:
            tshift_output[i] = tshift_input[i]
    #print(tshift_output[0:10,0:20])
        # Visualize result of subsample #
    corr = abs(np.fft.ifft2(np.fft.ifftshift(tshift_output)))
    corr -= corr.min()
    corr = corr / corr.max()
    corr = corr * 255.0
    corr += 0.5
    corr = corr.astype(int)
    return corr
    #return 20*np.log(abs(tshift_output))
