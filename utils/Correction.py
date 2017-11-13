# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:21:25 2017

@author: Alex
"""

import cv2
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

def Correction(subsampled_img, network_output, substep, lowfreqPercent):
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

        # 2-dimensional fast Fourier transform
    t_output = np.fft.fft2(network_output)#_slice)
    t_input = np.fft.fft2(subsampled_img)#_slice)

        # shifts 0 frequency to center
    tshift_output = np.fft.fftshift(t_output)
    tshift_input = np.fft.fftshift(t_input)

    mo = 20*np.log(np.abs(tshift_output.reshape(256,256)))
    mi = 20*np.log(np.abs(tshift_input.reshape(256,256)))
    plt.subplot(121),plt.imshow(mo, cmap = 'gray', vmin=0, vmax=255)
    plt.title('Ground_Truth'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(mi, cmap = 'gray',vmin=0, vmax=255)
    plt.title('After_Correction'), plt.xticks([]), plt.yticks([])
    plt.show()


        #Subsampler,
        #accounts for the double-counted lines
    lowfreqModifiedPercent = 1.0/float(substep)*lowfreqPercent+lowfreqPercent

    start = len(tshift_output)/2-int(lowfreqModifiedPercent*float(len(tshift_output)))
    end = len(tshift_output)/2+int(lowfreqModifiedPercent*float(len(tshift_output)))

    for i in range(0, start):
        if i % substep == 0:
            tshift_output[i] = tshift_input[i]
    for i in range (start, end):
        tshift_output[i] = tshift_input[i]
    for i in range (end, len(tshift_output)):
        if i % substep == 0:
            tshift_output[i] = tshift_input[i]

        # Visualize result of subsample #
    corr = abs(np.fft.ifft2(np.fft.ifftshift(tshift_output)))
    return corr
