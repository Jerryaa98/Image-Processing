import math

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def addSPnoise(im, p):
    sp_noise_im = im.copy()
    n = im.shape[0]*im.shape[1]
    sam = np.int(n*p/2)
    pepper = random.sample(range(n), sam)
    salt = random.sample(range(n), sam)
    for s in salt:
        sp_noise_im[s//im.shape[1]][s % im.shape[1]] = 255

    for pep in pepper:
        sp_noise_im[pep//im.shape[1]][pep % im.shape[1]] = 0

    return sp_noise_im.astype(np.uint8)


def addGaussianNoise(im, s):
    gaussian_noise_im = im.copy()
    gaussian_noise_im += np.random.normal(loc=0, scale=s, size=im.shape).astype(np.uint8)
    return gaussian_noise_im.astype(np.uint8)


def cleanImageMedian(im, radius):
    median_im = im.copy()
    for i in range(radius, im.shape[0]-radius-1):
        for j in range(radius, im.shape[1]-radius-1):
            arr = im[i-radius:i+radius+1, j-radius:j+radius+1]
            median_im[i][j] = np.int(np.median(arr))

    median_im = median_im[radius:im.shape[0]-radius, radius:im.shape[1]-radius]

    return median_im.astype(np.uint8)


def cleanImageMean(im, radius, maskSTD):

    filter = np.zeros((2*radius+1, 2*radius+1))
    for i in range(2*radius+1):
        for j in range(2*radius+1):
            filter[i][j] = np.exp(-((i-radius)**2 + (j-radius)**2)/(2*(maskSTD**2)))

    filter = filter / np.sum(filter)
    cleaned_im = convolve2d(im, filter, mode="same")
    return cleaned_im.astype(np.uint8)


def bilateralFilt(im, radius, stdSpatial, stdIntensity):
    bilateral_im = im.copy()

    gs = np.zeros((2*radius+1, 2*radius+1))
    for i in range(2*radius+1):
        for j in range(2*radius+1):
            gs[i][j] = np.exp(-(((i-radius)**2 + (j-radius)**2)/(2*(stdSpatial**2))))

    gs = gs / np.sum(gs)

    for i in range(radius, im.shape[0]-radius-1):
        for j in range(radius, im.shape[1]-radius-1):
            window = im[i-radius:i+radius+1, j-radius:j+radius+1]
            window = window.astype(np.float)
            gi = np.exp(-0.5*(((window-(im[i, j]))/stdIntensity)**2))
            gi = gi / np.sum(gi)
            gigs = cv2.multiply(gi, gs)
            #convolve2d(gs, gi, mode="same")
            bilateral_im[i, j] = (np.sum(cv2.multiply(gigs, window)))/np.sum(gigs)

    bilateral_im = bilateral_im[radius:im.shape[0]-radius, radius:im.shape[1]-radius]

    return bilateral_im.astype(np.uint8)


