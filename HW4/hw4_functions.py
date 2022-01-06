import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter

def print_IDs():
    print("211705538 + 318156171 \n")


def clean_im1(im):
    clean_im = im
    clean_im = median_filter(clean_im, size=3)
    dst = np.float32([[0, 0],
                      [255, 0],
                      [0, 255],
                      [255, 255]])

    src = np.float32([[6, 20],
                      [111, 20],
                      [6, 130],
                      [111, 130]])

    T = cv2.getPerspectiveTransform(src, dst)
    clean_im = cv2.warpPerspective(clean_im, T, (256, 256), flags=cv2.INTER_CUBIC)
    clean_im = contrastEnhance(clean_im, [0, 255])
    return clean_im.astype(np.uint8)


def clean_im2(im):
    img_fourier = np.fft.fftshift(np.fft.fft2(im))
    img_fourier[124][100] = -1
    img_fourier[132][156] = -1
    clean_im = abs(np.fft.ifft2(img_fourier))
    clean_im = contrastEnhance(clean_im, [0, 256])
    return clean_im.astype(np.uint8)


def clean_im3(im):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    clean_im = convolve2d(im, kernel, mode="same")
    clean_im = contrastEnhance(clean_im, [0, 256])
    return clean_im


def clean_im4(im):
    ffim = np.fft.fft2(im)
    kernel = np.zeros([256, 256])
    kernel[0][0] = 0.5
    kernel[4][79] = 0.5
    ffkernel = np.fft.fft2(kernel)
    ffkernel[ffkernel < 0.000005] = 1
    ffclean_im = ffim/ffkernel
    clean_im = abs(np.fft.ifft2(ffclean_im))
    clean_im = contrastEnhance(clean_im, [0, 256])
    return clean_im.astype(np.uint8)


def clean_im5(im):
    clean_im = im
    clean_im = median_filter(clean_im, size=(1, 5))
    clean_im = median_filter(clean_im, size=(3, 1))
    clean_im[0:90, 142:] = gaussian_filter(clean_im[0:90, 142:], sigma=(0, 7), order=0)
    clean_im[90:, :] = gaussian_filter(clean_im[90:, :], sigma=(0, 7), order=0)
    clean_im[0:90, 0:142] = im[0:90, 0:142]
    clean_im = contrastEnhance(clean_im, [0, 256])
    return clean_im


def clean_im6(im):
    img_fourier = np.fft.fftshift(np.fft.fft2(im))
    img_fourier[0:108, 0:256] /= 2
    img_fourier[147:256, 0:256] /= 2
    img_fourier[0:256, 0:108] /= 2
    img_fourier[0:256, 147:256] /= 2
    clean_im = abs(np.fft.ifft2(img_fourier))
    clean_im = contrastEnhance(clean_im, [0, 256])
    return clean_im.astype(np.uint8)


def clean_im7(im):
    kernel = np.zeros([191, 191])
    kernel[0:1, 0:10] = 1
    imfft = np.fft.fft2(im)
    kfft = np.fft.fft2(kernel)
    clean_im = abs(np.fft.ifft2(imfft/kfft))
    clean_im = contrastEnhance(clean_im, [0, 256])
    return clean_im.astype(np.uint8)


def clean_im8(im):
    clean_im = gamma_correction(im)
    clean_im = contrastEnhance(clean_im, [0, 255])
    return clean_im

#Helper functions from prev HWs

def contrastEnhance(im, range):
    nim = im
    maxi = np.max(np.ravel(nim))
    mini = np.min(np.ravel(nim))

    if maxi-mini == 0:
        b = range[1]
        a = 1
    else:
        b = np.float((np.float(maxi)*np.float(range[0]) - np.float(mini)*np.float(range[1])))/np.float((maxi-mini))
        a = np.float((range[1]-range[0]))/(maxi-mini)

    nim = nim * a
    nim = nim + b
    return nim


def sliceMat(im):
    Slices = np.zeros([256, len(im)*len(im[0])])
    for grayscale in range(256):
        Slices[grayscale] = np.ravel(im == grayscale)
    return Slices.transpose()


def mapImage(im, tm):
    slices = sliceMat(im).transpose()
    TMim = np.zeros([1, len(im)*len(im[0])])
    for gs in range(256):
        TMim[0] += tm[0][gs] * slices[gs]
    TMim = TMim.reshape((len(im), len(im[0])))
    return TMim


def gamma_correction(im):
    tm = np.zeros([1, 256])
    tm[0] = np.array(range(256))
    tm = tm/256
    tm = np.power(tm, 1/2.2)
    tm = tm*256
    corrected_im = mapImage(im, tm)
    return corrected_im
