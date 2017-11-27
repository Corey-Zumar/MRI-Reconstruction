# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:21:25 2017

@author: Alex
"""

import cv2
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

def subsample(analyze_img_path, substep=4, lowfreqPercent=0.04):
    """
    Subsamples an MRI image in Analyze 7.5 format
    Note: must have .hdr file

    Parameters
    ------------
    analyze_img_path : str
        The path to the Analyze image path (with ".img" extension)
    substep : int
        every substep-th line will be included (4 in paper)
    lowfrewPercent :  float
        percent of low frequencies to add into model (0.04 in paper)

    Returns
    ------------
    imgarr: numpy.core.memmap.memmap
    An numpy image object representing a list of subsampled human-
    interpretable images. Values are scaled to range from 0-255

    subsampled_img_K: numpy.core.memmap.memmap
    An numpy image object representing a list of subsampled images in k-space.
    These are complex numbers
    """
    #Load image
    img = nib.load(analyze_img_path)
    hdr = img.get_header()
    data = img.get_data()
    data = np.array(np.squeeze(img.get_data()), dtype=np.float32)
    data -= data.min()
    data = data / data.max()
    data = data * 255.0
    #data += 0.5

    subsampled_img_K = np.ones_like(data, dtype='complex')
    #data = data.astype(int)
    imgarr = np.ones_like(data)

    np.set_printoptions(threshold='nan')

    #iterate over each slice

    for slice in range(data.shape[2]):
        data_slice = np.squeeze(data[:,:,slice])


        # 2-dimensional fast Fourier transform
        t = np.fft.fft2(data_slice)

        # shifts 0 frequency to center
        tshift = np.fft.fftshift(t)

        # initialize a subsampled array with complex numbers
        subshift = np.ones_like(tshift)

        #Subsampler,
        #accounts for the double-counted lines
        lowfreqModifiedPercent = 1.0/float(substep)*lowfreqPercent+lowfreqPercent

        start = len(tshift)/2-int(lowfreqModifiedPercent*float(len(tshift)))
        end = len(tshift)/2+int(lowfreqModifiedPercent*float(len(tshift)))

        for i in range(0, start):
            if i % substep == 0:
                subshift[i] = tshift[i]
        for i in range (start, end):
            subshift[i] = tshift[i]
        for i in range (end, len(tshift)):
            if i % substep == 0:
                subshift[i] = tshift[i]

        # Visualize result of subsample #
        #print(slice)
        #print(type(imgarr))
        reconsubshift = abs(np.fft.ifft2(np.fft.ifftshift(subshift)).real).astype(float)
        #reconsubshift -= reconsubshift.min()
        #reconsubshift /= reconsubshift.max()
        #reconsubshift *= 255.0
        imgarr[:,:,slice] = reconsubshift

        subsampled_img_K[:,:,slice] = subshift
    #    if slice == 70:
    #        plt.subplot(121),plt.imshow(20*np.log(np.abs(subshift)), cmap='gray')
    #        plt.title('A        B            C          D'), plt.xticks([]), plt.yticks([])
    #        plt.subplot(122),plt.imshow(20*np.log(np.abs(tshift)), cmap = 'gray')
    #        plt.title('Subsampled'), plt.xticks([]), plt.yticks([])
    #        plt.show()
    #print(subsampled_img_K[:,:,70])
    return imgarr, subsampled_img_K
