import numpy as np
import cv2
import matplotlib.pyplot as plt

global frameCount
frameCount = 1

def writeMorphingVideo(image_list, video_name):
    out = cv2.VideoWriter(video_name+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 40.0, image_list[0].shape, 0)
    for im in image_list:
        out.write(im)
    out.release()


def createMorphSequence (im1, im1_pts, im2, im2_pts, t_list, transformType):
    global frameCount
    frameCount = 1

    I = np.identity(3)
    if transformType:
        T12 = findProjectiveTransform(im1_pts, im2_pts)
        T21 = findProjectiveTransform(im2_pts, im1_pts)
    else:
        T12 = findAffineTransform(im1_pts, im2_pts)
        T21 = findAffineTransform(im2_pts, im1_pts)
    ims = []
    for t in t_list:
        nim1 = mapImage(im1, (t*T12 + (1-t)*I), im1.shape)
        nim2 = mapImage(im2, ((1-t)*T21 + t*I), im2.shape)
        nim = nim1*(1-t) + t*nim2
        nim = np.round(nim)
        nim = nim.astype(np.uint8)
        nim = cv2.cvtColor(nim, cv2.COLOR_BGR2GRAY)
        ims.append(nim)

        print("frame {}".format(frameCount))
        frameCount += 1

    return ims


def mapImage(im, T, sizeOutIm):
    im_new = np.zeros(sizeOutIm)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # create meshgrid of all coordinates in new image [x,y]
    xx, yy = np.meshgrid(range(sizeOutIm[0]), range(sizeOutIm[1]))
    xx = xx.ravel()
    yy = yy.ravel()
    xy = np.vstack((xx, yy))
    homcoord = np.full((1, xy.shape[1]), 1, dtype=np.float)

    # add homogenous coord [x,y,1]
    xy1 = np.vstack((xy, homcoord))
    # calculate source coordinates that correspond to [x,y,1] in new image
    res = np.matmul(np.linalg.pinv(T), xy1)
    res[0] = np.divide(res[0], res[2])
    res[1] = np.divide(res[1], res[2])
    res[2] = np.divide(res[2], res[2])

    # find coordinates outside range and delete (in source and target)
    res = res.transpose()
    xy = xy.transpose()

    out1 = np.where((res[:, 0] >= sizeOutIm[0]-1) | (res[:, 0] < 0))[0]
    res = np.delete(res, out1, axis=0)
    xy = np.delete(xy, out1, axis=0)

    out2 = np.where((res[:, 1] >= sizeOutIm[1]-1) | (res[:, 1] < 0))[0]
    res = np.delete(res, out2, axis=0)
    xy = np.delete(xy, out2, axis=0)

    res = res.transpose()
    xy = xy.transpose()

    # interpolate - bilinear
    X = res[1]
    Y = res[0]
    x_left = (np.floor(X)).astype(np.int)
    x_right = (np.ceil(X)).astype(np.int)
    y_top = (np.floor(Y)).astype(np.int)
    y_bottom = (np.ceil(Y)).astype(np.int)
    upper_left = im[x_left, y_top]
    upper_right = im[x_right, y_top]
    bottom_left = im[x_left, y_bottom]
    bottom_right = im[x_right, y_bottom]
    deltaX = X - x_left
    deltaY = Y - y_top

    S = bottom_right * deltaX + bottom_left * (1 - deltaX)
    N = upper_right * deltaX + upper_left * (1 - deltaX)
    V = N * deltaY + S * (1 - deltaY)

    for i in range(xy.shape[1]):
        im_new[xy[1][i]][xy[0][i]] = V[i]

    return im_new.astype(np.uint8)

def findProjectiveTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[0]
    xtag = np.delete(pointsSet2, 2, 1)
    xtag = xtag.reshape(2 * N, 1)
    X = np.full((2 * N, 8), 0, dtype=np.float)

    j = 0
    # iterate over points to create x , x'
    for i in range(0, 2 * N, 2):
        X[i][0] = pointsSet1[j][0]
        X[i][1] = pointsSet1[j][1]
        X[i][4] = 1
        X[i][6] = -pointsSet1[j][0] * xtag[i][0]
        X[i][7] = -pointsSet1[j][1] * xtag[i][0]

        X[i + 1][2] = pointsSet1[j][0]
        X[i + 1][3] = pointsSet1[j][1]
        X[i + 1][5] = 1
        X[i + 1][6] = -pointsSet1[j][0] * xtag[i + 1][0]
        X[i + 1][7] = -pointsSet1[j][1] * xtag[i + 1][0]

        j += 1

    # calculate T - be careful of order when reshaping it
    T = np.matmul(np.linalg.pinv(X), xtag)
    T = T.reshape(8, 1)
    newrow = [1]
    T = np.vstack([T, newrow])
    T = T.reshape(3, 3)
    c = T[0][2]
    T[0][2] = T[1][1]
    T[1][1] = T[1][0]
    T[1][0] = c

    return T


def findAffineTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[0]
    xtag = np.delete(pointsSet2, 2, 1)
    xtag = xtag.reshape(2*N, 1)
    X = np.full((2*N, 6), 0, dtype=np.float)

    j = 0
    # iterate over points to create x , x'
    for i in range(0, 2*N, 2):
        X[i][0] = pointsSet1[j][0]
        X[i][1] = pointsSet1[j][1]
        X[i][4] = 1
        X[i+1][2] = pointsSet1[j][0]
        X[i+1][3] = pointsSet1[j][1]
        X[i+1][5] = 1
        j += 1

    # calculate T - be careful of order when reshaping it
    T = np.matmul(np.linalg.pinv(X), xtag)
    T = T.reshape(2, 3)
    c = T[0][2]
    T[0][2] = T[1][1]
    T[1][1] = T[1][0]
    T[1][0] = c
    newrow = [0, 0, 1]
    T = np.vstack([T, newrow])

    return T


def getImagePts(im1, im2, varName1, varName2, nPoints):

    plt.imshow(im1)
    imagePts1 = plt.ginput(n=nPoints, show_clicks=True, timeout=-1)
    plt.imshow(im2)
    imagePts2 = plt.ginput(n=nPoints, show_clicks=True, timeout=-1)

    imagePts1 = list(sum(imagePts1, ()))
    imagePts2 = list(sum(imagePts2, ()))
    imagePts1 = np.reshape(imagePts1, (nPoints, 2))
    imagePts2 = np.reshape(imagePts2, (nPoints, 2))

    imgPt1 = np.full((3, nPoints), 1, dtype=int)
    imgPt2 = np.full((3, nPoints), 1, dtype=int)

    imgPt1[0] = np.round(np.transpose(imagePts1)[0])
    imgPt1[1] = np.round(np.transpose(imagePts1)[1])
    imgPt2[0] = np.round(np.transpose(imagePts2)[0])
    imgPt2[1] = np.round(np.transpose(imagePts2)[1])

    imgPt1 = np.transpose(imgPt1)
    imgPt2 = np.transpose(imgPt2)

    np.save(varName1+".npy", imgPt1)
    np.save(varName2+".npy", imgPt2)

