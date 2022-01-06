import matplotlib.pyplot as plt
from hw2_functions import *
################################################################
#211705538
#318156171
################################################################
#A
################################################################
path_image = r'FaceImages\Face4.tif'
Face4 = cv2.imread(path_image)
path_image = r'FaceImages\Face5.tif'
Face5 = cv2.imread(path_image)
#getImagePts(Face4, Face5, "Face4Pts", "Face5Pts", 36)
imagePts1 = np.load("Face4Pts.npy")
imagePts2 = np.load("Face5Pts.npy")
#writeMorphingVideo(createMorphSequence(Face4, imagePts1, Face5, imagePts2, np.linspace(0, 1, 200), 0), "Face45Aff")
#writeMorphingVideo(createMorphSequence(Face4, imagePts1, Face5, imagePts2, np.linspace(0, 1, 200), 1), "Face45Proj")

###############################################################
#B
###############################################################
path_image = r'img\square.tif'
square = cv2.imread(path_image)
path_image = r'img\poly.tif'
poly = cv2.imread(path_image)

imagePts1 = np.load("sqrPts.npy")
imagePts2 = np.load("polyPts.npy")

#Convert square to poly
bAff = createMorphSequence(square, imagePts1, poly, imagePts2, [0.4], 0)[0]
bProj = createMorphSequence(square, imagePts1, poly, imagePts2, [0.4], 1)[0]

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(bAff, cmap='gray', vmin=0, vmax=255)
plt.title("Affine")
plt.subplot(1, 2, 2)
plt.imshow(bProj, cmap='gray', vmin=0, vmax=255)
plt.title("Projective")
plt.show()


################################
#C
################################
#C_i_###########################
path_image = r'FaceImages\Face3.tif'
Face3 = cv2.imread(path_image)
path_image = r'FaceImages\Face4.tif'
Face4 = cv2.imread(path_image)

#getImagePts(Face3, Face4, "Ci1low", "Ci2low", 3)
Ci1lowPts = np.load("Ci1low.npy")
Ci2lowPts = np.load("Ci2low.npy")

#getImagePts(Face3, Face4, "Ci1high", "Ci2high", 12)
Ci1highPts = np.load("Ci1high.npy")
Ci2highPts = np.load("Ci2high.npy")

CiLow = createMorphSequence(Face3, Ci1lowPts, Face4, Ci2lowPts, [0.5], 1)[0]
CiHigh = createMorphSequence(Face3, Ci1highPts, Face4, Ci2highPts, [0.5], 1)[0]

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(CiLow, cmap='gray', vmin=0, vmax=255)
plt.title("3 pts")
plt.subplot(1, 2, 2)
plt.imshow(CiHigh, cmap='gray', vmin=0, vmax=255)
plt.title("12 pts")
plt.show()

#C_ii_####################################

#getImagePts(Face3, Face4, "Cii1foc", "Cii2foc", 12)
Cii1focPts = np.load("Cii1foc.npy")
Cii2focPts = np.load("Cii2foc.npy")

CiFoc = createMorphSequence(Face3, Cii1focPts, Face4, Cii2focPts, [0.5], 1)[0]
CiDis = CiHigh

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(CiFoc, cmap='gray', vmin=0, vmax=255)
plt.title("focused")
plt.subplot(1, 2, 2)
plt.imshow(CiDis, cmap='gray', vmin=0, vmax=255)
plt.title("distributed")
plt.show()

####################################
#D) from human to statue
####################################

path_image = r'img\fayroz2.tif'
frzR = cv2.imread(path_image)
path_image = r'img\frzStat.tif'
frzS = cv2.imread(path_image)
#getImagePts(frzR, frzS, "frzR", "frzS", 36)
imagePts1 = np.load("frzR.npy")
imagePts2 = np.load("frzS.npy")
#writeMorphingVideo(createMorphSequence(frzR, imagePts1, frzS, imagePts2, np.linspace(0, 1, 200), 0), "frzAff")
#writeMorphingVideo(createMorphSequence(frzR, imagePts1, frzS, imagePts2, np.linspace(0, 1, 200), 1), "frzProj")

