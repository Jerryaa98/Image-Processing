import cv2
import numpy as np
import matplotlib.pyplot as plt


def print_IDs():
	#print("123456789")
    print("211705538+318156171\n")


def contrastEnhance(im,range):
    nim = im
    a = (255 / (range[1] - range[0]))
    b = - ((255 * range[0]) / (range[1] - range[0]))
    nim = nim * a
    nim = nim + b
    return nim, a, b


def showMapping(old_range, a, b):
    imMin = np.min(old_range)
    imMax = np.max(old_range)
    x = np.arange(imMin, imMax+1, dtype=np.float)
    y = a * x + b
    plt.figure()
    plt.plot(x, y)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.title('contrast enhance mapping')


def minkowski2Dist(im1,im2):
    h1, _ = np.histogram(im1, bins=256, range=(0, 255), density=True)
    h2, _ = np.histogram(im2, bins=256, range=(0, 255), density=True)
    diff = h1 - h2
    diff = np.power(diff, 2)
    d = np.sum(diff)
    d = d**0.5
    return d


def meanSqrDist(im1, im2):
    d = np.float((np.square(im1 - im2)).mean())
    return d


def sliceMat(im):
    Slices = np.zeros([256, len(im)*len(im[0])])
    for grayscale in range(256):
        Slices[grayscale] = np.ravel(im == grayscale)
    return Slices.transpose()


def SLTmap(im1, im2):
    slices =  sliceMat(im1)
    slices_t = slices.transpose()
    flatIm2 = np.zeros([1, len(im1)*len(im1[0])])
    flatIm2[0] = im2.reshape(-1)
    TM = np.matmul(flatIm2, slices)
    for gs in range(0, 256):
        if np.sum(slices_t[gs]) != 0:
           TM[0][gs] = TM[0][gs] / np.sum(slices_t[gs])
    return mapImage(im1, TM), TM


def mapImage(im, tm):
    slices = sliceMat(im).transpose()
    TMim = np.zeros([1, len(im)*len(im[0])])
    for gs in range(256):
        TMim[0] += tm[0][gs] * slices[gs]
    TMim = TMim.reshape((len(im), len(im[0])))
    return TMim


def sltNegative(im):
    tm = np.zeros([1,256])
    tm[0] = np.array(range(256, 0, -1))
    nim = mapImage(im, tm)
    return nim


def sltThreshold(im, thresh):
    tm = np.zeros([1, 256])
    tm[0][thresh:] = 255
    nim = mapImage(im, tm)
    return nim
