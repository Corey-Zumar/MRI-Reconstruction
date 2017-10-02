# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:21:25 2017

@author: Alex
"""

import cv2
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

####### Important variables ####
substep = 4 #every substep-th line is kept
lowfreqPercent = 0.04 #% of low frequencies to add in
#please note the double-counted lines HAVE been accounted for


#######
# input image
#img = cv2.imread('IMG1.png',0)

img = nib.load('img2.img')
hdr = img.get_header()
data = img.get_data()

#need to iterate over all images here
data = np.squeeze(data[:,:,70])
plt.imshow(data, cmap='gray')
plt.show()


# 2-dimensional fast Fourier transform
t = np.fft.fft2(data)

# shifts 0 frequency to center
tshift = np.fft.fftshift(t)

##### Block to compare 2 plots #####

# tshift is complex: a + ib
# abs returns sqrt(a^2+b^2)
# I THINK THE 20 IS ARBITRARY
magnitude_spectrum = 20*np.log(np.abs(tshift))
plt.subplot(121),plt.imshow(data, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
##### Block to compare 2 plots #####

# initialize a subsampled array with complex numbers
subshift = np.ones_like(tshift)

# Add in low frequency data.
# Might need to consider the amount of data included here that would have
# already been included in the 4-skip step.

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

##### Block to compare 2 plots #####

# tshift is complex: a + ib
# abs returns sqrt(a^2+b^2)
# I THINK THE 20 IS ARBITRARY
magnitude_spectrum = 20*np.log(np.abs(subshift))
plt.subplot(121),plt.imshow(data, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Subsampled Image'), plt.xticks([]), plt.yticks([])
plt.show()
##### Block to compare 2 plots #####

# Visualize result of subsample #
reconshift = abs(np.fft.ifft2(tshift))
reconsubshift = abs(np.fft.ifft2(subshift))

##### Block to compare 2 plots #####
plt.subplot(121),plt.imshow(reconshift, cmap = 'gray')
plt.title('iFFT(input)'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(reconsubshift, cmap = 'gray')
plt.title('iFFT(subsampled)'), plt.xticks([]), plt.yticks([])
plt.show()
##### Block to compare 2 plots #####
