from hw4_functions import *


if __name__ == "__main__":
    print("----------------------------------------------------\n")
    print_IDs

    print("-----------------------image 1----------------------\n")
    im1 = cv2.imread(r'Images\baby.tif')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1_clean = clean_im1(im1)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
    plt.title("Baby")
    plt.subplot(1, 2, 2)
    plt.imshow(im1_clean, cmap='gray', vmin=0, vmax=255)
    plt.title("Cleaned Baby")

    print("Describe the problem with the image and your method/solution: \n")
    print("The photo is duplicated 3 times and it had SP noise,"
          "\nI removed the SP noise with median filter, "
          "\nthen used proj transform to resize one of the 3 photos to the full size \n")
    
    print("-----------------------image 2----------------------\n")
    im2 = cv2.imread(r'Images\windmill.tif')
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im2_clean = clean_im2(im2)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.title("Windmill")
    plt.subplot(1, 2, 2)
    plt.imshow(im2_clean, cmap='gray', vmin=0, vmax=255)
    plt.title("Cleaned Windmill")

    print("Describe the problem with the image and your method/solution: \n")
    print("The photo had frequent noise, "
          "\nso I moved to the freq domain and searched for unrealistic high coefficient and zeroed it,"
          "\nthen moved back to the the spatial domain\n")
    

    print("-----------------------image 3----------------------\n")
    im3 = cv2.imread(r'Images\watermelon.tif')
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
    im3_clean = clean_im3(im3)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im3, cmap='gray', vmin=0, vmax=255)
    plt.title("Watermelon")
    plt.subplot(1, 2, 2)
    plt.imshow(im3_clean, cmap='gray', vmin=0, vmax=255)
    plt.title("Sharpened Watermelon")

    print("Describe the problem with the image and your method/solution: \n")
    print("The photo had been blurred, so I sharpened the photo using a sharpening mask \n")
    

    print("-----------------------image 4----------------------\n")
    im4 = cv2.imread(r'Images\umbrella.tif')
    im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)
    im4_clean = clean_im4(im4)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im4, cmap='gray', vmin=0, vmax=255)
    plt.title("Umbrella")
    plt.subplot(1, 2, 2)
    plt.imshow(im4_clean, cmap='gray', vmin=0, vmax=255)
    plt.title("Cleaned Umbrella")

    print("Describe the problem with the image and your method/solution: \n")
    print("The photo was echoed, I calculated the mask that was used to do this,"
          "\nthen moved to the freq domain and divided the fft of the img by the fft of the mask,"
          "\nthen moved back to the spatial domain\n")

    
    print("-----------------------image 5----------------------\ n")
    im5 = cv2.imread(r'Images\USAflag.tif')
    im5 = cv2.cvtColor(im5, cv2.COLOR_BGR2GRAY)
    im5_clean = clean_im5(im5)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im5, cmap='gray', vmin=0, vmax=255)
    plt.title("USA Flag")
    plt.subplot(1, 2, 2)
    plt.imshow(im5_clean, cmap='gray', vmin=0, vmax=255)
    plt.title("Cleaned USA Flag")

    print("Describe the problem with the image and your method/solution: \n")
    print("Someone wrote stuff on the photo and we wanted to remove it, "
          "\nwe used the median filter to remove the writing and gaussian blur to make the image smother,"
          "\nall of this was only on the part without the stars within the flag "
          "\nand only in the y-axis direction \n")

    

    print("-----------------------image 6----------------------\ n")
    im6 = cv2.imread(r'Images\cups.tif')
    im6 = cv2.cvtColor(im6, cv2.COLOR_BGR2GRAY)
    im6_clean = clean_im6(im6)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im6, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im6_clean, cmap='gray', vmin=0, vmax=255)

    print("Describe the problem with the image and your method/solution: \n")
    print("The photo had kind of ringing, "
          "\nto clean it we moved to the freq domain and divided the high frequencies by 2\n")
    
    print("-----------------------image 7----------------------\ n")
    im7 = cv2.imread(r'Images\house.tif')
    im7 = cv2.cvtColor(im7, cv2.COLOR_BGR2GRAY)
    im7_clean = clean_im7(im7)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im7, cmap='gray', vmin=0, vmax=255)
    plt.title("House")
    plt.subplot(1, 2, 2)
    plt.imshow(im7_clean, cmap='gray', vmin=0, vmax=255)
    plt.title("Cleaned House")

    print("Describe the problem with the image and your method/solution: \n")
    print("The photo had motion blur, to remove it we calculated the mask that was used "
          "\nto do it and divided the image by it in the freq domain and moved back to the spatial domain \n")

    
    print("-----------------------image 8----------------------\ n")
    im8 = cv2.imread(r'Images\bears.tif')
    im8 = cv2.cvtColor(im8, cv2.COLOR_BGR2GRAY)
    im8_clean = clean_im8(im8)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im8, cmap='gray', vmin=0, vmax=255)
    plt.title("Bears")
    plt.subplot(1, 2, 2)
    plt.imshow(im8_clean, cmap='gray', vmin=0, vmax=255)
    plt.title("Cleaned Bears")

    print("Describe the problem with the image and your method/solution: \n")
    print("The photo was really dark, to enlighten it we did gamma correction \n")

    print("------------------------------------------------------------------- \n")
    print("All photos were contrast enhanced as well \n")

    plt.show()
