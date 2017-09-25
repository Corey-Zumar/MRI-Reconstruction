# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:21:25 2017

@author: Alex
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# input image
img = cv2.imread('IMG1.png',0)

# 2-dimensional fast Fourier transform
t = np.fft.fft2(img)

# shifts 0 frequency to center
tshift = np.fft.fftshift(t)


##### Block to compare 2 plots #####

# tshift is complex: a + ib
# abs returns sqrt(a^2+b^2)
# I THINK THE 20 IS ARBITRARY
magnitude_spectrum = 20*np.log(np.abs(tshift))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
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
midpercent = .04

start = len(tshift)/2-int(midpercent*float(len(tshift)))
end = len(tshift)/2+int(midpercent*float(len(tshift)))

for i in range(0, start):
    if i % 4 == 0:
        subshift[i] = tshift[i]
for i in range (start, end):
    subshift[i] = tshift[i]
for i in range (end, len(tshift)):
    if i % 4 == 0:
        subshift[i] = tshift[i]

##### Block to compare 2 plots #####

# tshift is complex: a + ib
# abs returns sqrt(a^2+b^2)
# I THINK THE 20 IS ARBITRARY
magnitude_spectrum = 20*np.log(np.abs(subshift))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
##### Block to compare 2 plots #####
