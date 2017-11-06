# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:21:25 2017

@author: Alex
"""

import cv2
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

def subsample(subsampled_img, network_output, substep, lowfreqPercent):
    """
    Corrects network output using the input subsampled image.

    Parameters
    ------------
    subsampled_img : numpy.core.memmap.memmap
        Subsampled image used as network input
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

    for slice in range(network_output.shape[2]):
        network_output_slice = np.squeeze(network_output[:,:,slice])
        subsampled_slice = np.squeeze(subsampled_slice[:,:,slice])

        # 2-dimensional fast Fourier transform
        t_output = np.fft.fft2(network_output_slice)
        t_input = np.fft.fft2(subsampled_slice)

        # shifts 0 frequency to center
        tshift_output = np.fft.fftshift(t_output)
        tshift_input = np.fft.fftshift(t_input)

        #Subsampler,
        #accounts for the double-counted lines
        lowfreqModifiedPercent = 1.0/float(substep)*lowfreqPercent+lowfreqPercent

        start = len(tshift)/2-int(lowfreqModifiedPercent*float(len(tshift)))
        end = len(tshift)/2+int(lowfreqModifiedPercent*float(len(tshift)))

        for i in range(0, start):
            if i % substep == 0:
                tshift_output[i] = tshift_input[i]
        for i in range (start, end):
            tshift_output[i] = tshift_input[i]
        for i in range (end, len(tshift)):
            if i % substep == 0:
                tshift_output[i] = tshift_input[i]

        # Visualize result of subsample #
        corr = abs(np.fft.ifft2(tshift_output))
        imgarr[:,:,slice,0] = corr
        print(slice)
        print(type(imgarr))

    return np.squeeze(imgarr)
