from hw3_functions import *

if __name__ == "__main__":
    print("211705538_318156171")

    # feel free to load different image than lena
    lena = cv2.imread(r"Images\lena.tif")
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

    # 1 ----------------------------------------------------------
    # add salt and pepper noise - low
    lena_sp_low = addSPnoise(lena_gray, 0.02)

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(lena_sp_low, cmap='gray', vmin=0, vmax=255)
    plt.title("salt and pepper - low")
    plt.subplot(2, 3, 4)
    plt.imshow(cleanImageMedian(lena_sp_low, 1), cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(cleanImageMean(lena_sp_low, 3, 10), cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(bilateralFilt(lena_sp_low, 3, 10, 10), cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")

    print("Conclusions -----  Median filter is best for low SP noise, then gaussian blur and finally BLF\n")

    # 2 ----------------------------------------------------------
    # add salt and pepper noise - high
    lena_sp_high = addSPnoise(lena_gray, 0.15)  # TODO - add low noise

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(lena_sp_high, cmap='gray', vmin=0, vmax=255)
    plt.title("salt and pepper - High")
    plt.subplot(2, 3, 4)
    plt.imshow(cleanImageMedian(lena_sp_high, 3), cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(cleanImageMean(lena_sp_high, 4, 20), cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(bilateralFilt(lena_sp_high, 4, 20, 15), cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")

    print("Conclusions -----  Median filter is best for low / high SP noise, then gaussian blur and finally BLF \n")

    # 3 ----------------------------------------------------------
    # add gaussian noise - low
    lena_gaussian = addGaussianNoise(lena_gray, 10)  # TODO - add low noise

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(lena_gaussian, cmap='gray', vmin=0, vmax=255)
    plt.title("gaussian noise - low")
    plt.subplot(2, 3, 4)
    plt.imshow(cleanImageMedian(lena_gaussian, 3), cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(cleanImageMean(lena_gaussian, 3, 10), cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(bilateralFilt(lena_gaussian, 3, 10, 15), cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")

    print("Conclusions -----  Bilateral filter is best for low gaussian noise, then Median and finally GB \n")

    # 4 ----------------------------------------------------------
    # add gaussian noise - high
    lena_gaussian = addGaussianNoise(lena_gray, 25)  # TODO - add high noise

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(lena_gaussian, cmap='gray', vmin=0, vmax=255)
    plt.title("gaussian noise - high")
    plt.subplot(2, 3, 4)
    plt.imshow(cleanImageMedian(lena_gaussian, 3), cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(cleanImageMean(lena_gaussian, 4, 25), cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(bilateralFilt(lena_gaussian, 4, 25, 25), cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")

    print("Conclusions -----  Gaussian blur is best for high gaussian noise, then BLF and finally median \n")

    plt.show()