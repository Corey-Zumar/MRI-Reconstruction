# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:21:25 2017

@author: Alex
"""

import numpy as np


def correct_output(subsampled_img_k,
                   network_output,
                   substep=4,
                   low_freq_percent=0.04):
    """
    Corrects network output using the input subsampled image.

    Parameters
    ------------
    subsampled_img_k : numpy.core.memmap.memmap
        The k-space subsampled image used to generate network input
    network_output: numpy.core.memmap.memmap
        Output from the CNN
    substep: int
        every substep-th line will be included (4 in paper)
    low_freq_percent :  float
        percent of low frequencies to add into model (0.04 in paper)

    Returns
    ------------
    numpy.core.memmap.memmap
    An numpy image object representing a list of subsampled human-
    interpretable images.
    """

    # 2-dimensional fast Fourier transform
    t_output = np.fft.fft2(network_output)

    # shifts zero frequency to center
    tshift_output = np.fft.fftshift(t_output)
    tshift_input = subsampled_img_k

    mod_low_freq_percent = 1.0 / float(
        substep) * low_freq_percent + low_freq_percent

    start = len(tshift_output) / 2 - int(
        mod_low_freq_percent * float(len(tshift_output)))
    end = len(tshift_output) / 2 + int(
        mod_low_freq_percent * float(len(tshift_output)))

    for i in range(0, start):
        if i % substep == 0:
            tshift_output[i] = tshift_input[i]
    for i in range(start, end):
        tshift_output[i] = tshift_input[i]
    for i in range(end, len(tshift_output)):
        if i % substep == 0:
            tshift_output[i] = tshift_input[i]

    corr = abs(np.fft.ifft2(np.fft.ifftshift(tshift_output)))
    corr -= corr.min()
    corr = corr / corr.max()
    corr = corr * 255.0
    corr = corr.astype(int)
    return corr
