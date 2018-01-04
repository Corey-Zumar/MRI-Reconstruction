# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:21:25 2017

@author: Alex
"""

import numpy as np


def subsample(analyze_img_data, substep=4, low_freq_percent=0.04):
    """
    Subsamples an MRI image in Analyze 7.5 format

    Parameters
    ------------
    analyze_img_data : np.ndarray
        A numpy representation of cropped and scaled Analyze image
        data with datatype `np.float32`
    substep : int
        every substep-th line will be included (4 in paper)
    low_freq_percent :  float
        percent of low frequencies to add into model (0.04 in paper)

    Returns
    ------------
    imgarr : numpy.core.memmap.memmap
    An numpy image object representing a list of subsampled human-
    interpretable images. Values are scaled to range from 0-255

    subsampled_img_k : numpy.core.memmap.memmap
    An numpy image object representing a list of subsampled images in k-space.
    These are complex numbers
    """

    data = np.copy(analyze_img_data)

    subsampled_img_k = np.ones_like(data, dtype='complex')
    imgarr = np.ones_like(data)

    np.set_printoptions(threshold='nan')

    for slice_idx in range(data.shape[2]):
        data_slice = np.squeeze(data[:, :, slice_idx])

        # 2-dimensional fast Fourier transform
        t = np.fft.fft2(data_slice)

        # shift zero frequency to center
        tshift = np.fft.fftshift(t)

        # initialize a subsampled array with complex numbers
        subshift = np.ones_like(tshift)

        #Subsampler,
        #accounts for the double-counted lines
        mod_low_freq_percent = 1.0 / float(
            substep) * low_freq_percent + low_freq_percent

        start = len(tshift) / 2 - int(
            mod_low_freq_percent * float(len(tshift)))
        end = len(tshift) / 2 + int(mod_low_freq_percent * float(len(tshift)))

        for i in range(0, start):
            if i % substep == 0:
                subshift[i] = tshift[i]
        for i in range(start, end):
            subshift[i] = tshift[i]
        for i in range(end, len(tshift)):
            if i % substep == 0:
                subshift[i] = tshift[i]

        reconsubshift = abs(np.fft.ifft2(
            np.fft.ifftshift(subshift)).real).astype(float)
        imgarr[:, :, slice_idx] = reconsubshift

        subsampled_img_k[:, :, slice_idx] = subshift

    return imgarr, subsampled_img_k
