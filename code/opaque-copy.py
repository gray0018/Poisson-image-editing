# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

def fix_source(source, mask, shape, offset):
    mydict = {}
    counter = 0

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j]>127:
                mydict[(i+offset[0], j+offset[1])] = counter
                counter += 1
    fixed_source = np.zeros(shape, dtype=int) #use int to avoid overflow
    fixed_source[max(0, offset[0]):min(source.shape[0]+offset[0], shape[0]), max(0, offset[1]):min(source.shape[1]+offset[1],shape[1]),:]=source[max(0,-offset[0]):min(source.shape[0], shape[0]-offset[0]),max(0,-offset[1]):min(source.shape[1], shape[1]-offset[1]),:]

    return fixed_source, mydict

offset = [[210, 10], [10, 28], [140, 80], [-40, 90], [60, 100], [-28, 88]]


for pic_index in range(1, 2):
    mask = cv2.imread("../data/mask_0{0}.jpg".format(pic_index), 0)
    source = cv2.imread("../data/source_0{0}.jpg".format(pic_index))
    target = cv2.imread("../data/target_0{0}.jpg".format(pic_index))

    fixed_source, D = fix_source(source, mask, target.shape, offset[pic_index-1]) #fixed source, same size with target

    A = np.zeros((len(D),len(D)), dtype=int)
    b = np.zeros((len(D),3), dtype=int)


    for k, v in D.items():
        A[v][v] = 4
        b[v] += 4*fixed_source[k[0]][k[1]] \
            - fixed_source[k[0]+1][k[1]] \
            - fixed_source[k[0]-1][k[1]] \
            - fixed_source[k[0]][k[1]+1] \
            - fixed_source[k[0]][k[1]-1]

        if (k[0]+1, k[1]) in D: # in D means this pixel is waiting to be calculated
            A[v][D[(k[0]+1, k[1])]] = -1
        else:
            b[v] += target[k[0]+1][k[1]]

        if (k[0]-1, k[1]) in D:
            A[v][D[(k[0]-1, k[1])]] = -1
        else:
            b[v] += target[k[0]-1][k[1]]

        if (k[0], k[1]+1) in D:
            A[v][D[(k[0], k[1]+1)]] = -1
        else:
            b[v] += target[k[0]][k[1]+1]

        if (k[0], k[1]-1) in D:
            A[v][D[(k[0], k[1]-1)]] = -1
        else:
            b[v] += target[k[0]][k[1]-1]

    A = sparse.csr_matrix(A)
    # b = sparse.csr_matrix(b)

    channel0 = lsqr(A, b[:,0])[0]
    channel1 = lsqr(A, b[:,1])[0]
    channel2 = lsqr(A, b[:,2])[0]

    channel0 = channel0.reshape([channel0.shape[0], 1])
    channel1 = channel1.reshape([channel1.shape[0], 1])
    channel2 = channel2.reshape([channel2.shape[0], 1])

    x = np.concatenate([channel0, channel1, channel2], axis=1)

    for k, v in D.items():
        if x[v][0]>255:
            target[k[0]][k[1]][0] = np.uint8(255)
        elif x[v][0]<0:
            target[k[0]][k[1]][0] = np.uint8(0)
        else:
            target[k[0]][k[1]][0] = np.uint8(round(x[v][0]))

        if x[v][1]>255:
            target[k[0]][k[1]][1] = np.uint8(255)
        elif x[v][1]<0:
            target[k[0]][k[1]][1] = np.uint8(0)
        else:
            target[k[0]][k[1]][1] = np.uint8(round(x[v][1]))

        if x[v][2]>255:
            target[k[0]][k[1]][2] = np.uint8(255)
        elif x[v][2]<0:
            target[k[0]][k[1]][2] = np.uint8(0)
        else:
            target[k[0]][k[1]][2] = np.uint8(round(x[v][2]))


        # target[k[0]][k[1]][0] = np.uint8(round(x[v][0])%256)
        # target[k[0]][k[1]][1] = np.uint8(round(x[v][1])%256)
        # target[k[0]][k[1]][2] = np.uint8(round(x[v][2])%256)

    cv2.imwrite("result_0{0}.jpg".format(pic_index), target)
