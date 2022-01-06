import numpy as np

from hw1_functions import *


if __name__ == "__main__":
    # feel free to add/remove/edit lines
	
    path_image = r'Images\darkimage.tif'
    darkimg = cv2.imread(path_image)
    darkimg_gray = cv2.cvtColor(darkimg, cv2.COLOR_BGR2GRAY)

    print("Start running script  ------------------------------------\n")
    print_IDs()

    print("a ------------------------------------\n")
    enhanced_img, a, b = contrastEnhance(darkimg_gray, [(np.ravel(darkimg_gray)).min(), (np.ravel(darkimg_gray)).max()])#add parameters

    # display images
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(darkimg)
    plt.title('original')

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_img, cmap='gray', vmin=0, vmax=255)
    plt.title('enhanced contrast')

    # print a,b
    print("a = {}, b = {}\n".format(a, b))

    #display mapping
    showMapping([(np.ravel(darkimg_gray)).min(), (np.ravel(darkimg_gray)).max()], a, b)#add parameters

    print("b ------------------------------------\n")
    enhanced2_img, a, b = contrastEnhance(enhanced_img, [(np.ravel(enhanced_img)).min(), (np.ravel(enhanced_img)).max()])#add parameters
    # print a,b
    print("enhancing an already enhanced image\n")
    print("a = {}, b = {}\n".format(a, b))

    # display the difference between the two image (Do not simply display both images)

    if (enhanced2_img-enhanced_img).all() == False:
        print("There is NO difference between the images")
    else:
        print("There is difference between the images")


    print("c ------------------------------------\n")
    d = minkowski2Dist(darkimg_gray, darkimg_gray)  #add parameters
    print("Minkowski distance between image and itself is: d = {}\n".format(d))

    path_image = r'Images\cups.tif'
    cups = cv2.imread(path_image)
    cups_gray = cv2.cvtColor(cups, cv2.COLOR_BGR2GRAY)
    minC = cups_gray.min()
    maxC = cups_gray.max()
    tm = np.zeros([1, 256])
    contrast = np.arange(minC, maxC, ((maxC-minC)/20), dtype=np.float)
    dists = np.zeros(20)
    # TODO: implement the loop that calculates minkowski distance as function of increasing contrast
    for k in range(0, 20):
        tm[0] = np.array(range(256))
        tm = tm * ((k+1)/20) #a
        tm = tm + ((1-((k+1)/20))*minC) #b
        nim = mapImage(cups_gray, tm)
        dists[k] = minkowski2Dist(cups_gray, nim)
        # if you want to see the difference between the 20 images
        if k % 5 == 0:
            plt.figure()
        plt.subplot(1, 5, (k % 5)+1)
        plt.imshow(nim, cmap='gray', vmin=0, vmax=255)
        plt.title("k={}".format(k+1))

    plt.figure()
    plt.plot(contrast, dists)
    plt.xlabel("contrast")
    plt.ylabel("distance")
    plt.title("Minkowski distance as function of contrast")

    print("d ------------------------------------\n")

    path_image = r'Images\barbarasmall.tif'
    barbra = cv2.imread(path_image)
    barbra_gray = cv2.cvtColor(barbra, cv2.COLOR_BGR2GRAY)
    tm = np.zeros([1, 256])
    tm[0] = np.array(range(0, 256))
    d = meanSqrDist(barbra_gray, mapImage(barbra_gray, tm))
    print("d = {}".format(d))
    # we use mapImage cause it slices our image and then maps each gray tone to a given tone
    # in this case we want to map each tone to itself.
    # then we prove that the 2 images are the same image by calculating MSE.


    print("e ------------------------------------\n")

    barbra_enhanced, _, _ = contrastEnhance(barbra_gray, [(np.ravel(barbra_gray)).min(), (np.ravel(barbra_gray)).max()])
    _ , barbra_en_tm = SLTmap(barbra_gray, barbra_enhanced)
    barbra_mul_255 = mapImage(barbra_gray, barbra_en_tm)
    d = meanSqrDist(barbra_gray, barbra_mul_255)
    print("sum of diff between image and slices*[0..255] = {}".format(d))
    d = meanSqrDist(barbra_enhanced, barbra_mul_255)
    print("sum of diff between contrast enhanced image and slices*[0..255] = {}".format(d))


    # then display
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(barbra)
    plt.title("original image")
    plt.subplot(1, 2, 2)
    plt.imshow(barbra_mul_255, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")



    print("f ------------------------------------\n")
    negative_im = sltNegative(darkimg_gray)
    plt.figure()
    plt.imshow(negative_im, cmap='gray', vmin=0, vmax=255)
    plt.title("negative image using SLT")



    print("g ------------------------------------\n")
    thresh = 120 # play with it to see changes
    lena = cv2.imread(r"Images\\RealLena.tif")
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    thresh_im = sltThreshold(lena_gray, thresh)#add parameters
	
    plt.figure()
    plt.imshow(thresh_im, cmap='gray', vmin=0, vmax=255)
    plt.title("thresh image using SLT")



    print("h ------------------------------------\n")
    im1 = lena_gray
    im2 = barbra_gray
    SLTim, _ = SLTmap(im1, im2)
	
    # then print
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(lena)
    plt.title("original image")
    plt.subplot(1, 3, 2)
    plt.imshow(SLTim, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")
    plt.subplot(1, 3, 3)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped to")

    d1 = meanSqrDist(im1, im2)
    d2 = meanSqrDist(SLTim, im2)
    print("mean sqr dist between im1 and im2 = {}\n".format(d1))
    print("mean sqr dist between mapped image and im2 = {}\n".format(d2))

    print("i ------------------------------------\n")
    # prove comutationally
    SLTim2, _ = SLTmap(im2, im1)
    d = meanSqrDist(im1, SLTim2)
    print(" {}".format(d))
    if d2 == d:
        print("SLTmapping is symmetric")
    else:
        print("SLTmapping is NOT symmetric")
    plt.show()