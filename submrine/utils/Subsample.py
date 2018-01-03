# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:21:25 2017

@author: Alex
"""

import cv2
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

TARGET_SLICE_WIDTH = 256
TARGET_SLICE_HEIGHT = 256

def center_crop(img_data):
    slice_width, slice_height, _ = img_data.shape()
    if (slice_width < TARGET_SLICE_WIDTH) or (slice_height < TARGET_SLICE_HEIGHT):
        raise Exception("The width and height of each MRI image slice must be at least 256 pixels!")

    width_crop = (slice_width - TARGET_SLICE_WIDTH) // 2
    height_crop = (slice_height - TARGET_SLICE_HEIGHT) // 2

    return img_data[width_crop:-width_crop,height_crop:-height_crop,:]

def subsample(analyze_img_path, substep=4, low_freq_percent=0.04):
    """
    Subsamples an MRI image in Analyze 7.5 format
    Note: must have .hdr file

    Parameters
    ------------
    analyze_img_path : str
        The path to the Analyze image path (with ".img" extension)
    substep : int
        every substep-th line will be included (4 in paper)
    low_freq_percent :  float
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

    img = nib.load(analyze_img_path)
    data = img.get_data()
    data = center_crop(data)
    data = np.array(np.squeeze(img.get_data()), dtype=np.float32)
    data = data[63:319,63:319,:]
    data -= data.min()
    data = data / data.max()
    data = data * 255.0
  
    subsampled_img_K = np.ones_like(data, dtype='complex')
    imgarr = np.ones_like(data)

    np.set_printoptions(threshold='nan')

    for slice_idx in range(data.shape[2]):
        data_slice = np.squeeze(data[:,:,slice_idx])

        # 2-dimensional fast Fourier transform
        t = np.fft.fft2(data_slice)

        # shift zero frequency to center
        tshift = np.fft.fftshift(t)

        # initialize a subsampled array with complex numbers
        subshift = np.ones_like(tshift)

        #Subsampler,
        #accounts for the double-counted lines
        mod_low_freq_percent = 1.0 / float(substep) * low_freq_percent + low_freq_percent

        start = len(tshift)/2-int(mod_low_freq_percent*float(len(tshift)))
        end = len(tshift)/2+int(mod_low_freq_percent*float(len(tshift)))

        for i in range(0, start):
            if i % substep == 0:
                subshift[i] = tshift[i]
        for i in range (start, end):
            subshift[i] = tshift[i]
        for i in range (end, len(tshift)):
            if i % substep == 0:
                subshift[i] = tshift[i]

        reconsubshift = abs(np.fft.ifft2(np.fft.ifftshift(subshift)).real).astype(float)
        imgarr[:,:,slice_idx] = reconsubshift

        subsampled_img_K[:,:,slice_idx] = subshift
    return imgarr, subsampled_img_K
