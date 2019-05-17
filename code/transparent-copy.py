# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy.sparse.linalg import cg

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


for pic_index in range(1, 6):
    mask = cv2.imread("../data/mask_0{0}.jpg".format(pic_index), 0)
    source = cv2.imread("../data/source_0{0}.jpg".format(pic_index))
    target = cv2.imread("../data/target_0{0}.jpg".format(pic_index))

    fixed_source, D = fix_source(source, mask, target.shape, offset[pic_index-1]) #fixed source, same size with target

    A = np.zeros((len(D),len(D)), dtype=int)
    b = np.zeros((len(D),3), dtype=int)

    for k, v in D.items():
        A[v][v] = 4
        for j in (1, -1):

            if (k[0]+j, k[1]) in D: # in D means this pixel is waiting to be calculated
                A[v][D[(k[0]+j, k[1])]] = -1
            else:
                b[v] += target[k[0]+j][k[1]]

            if (k[0], k[1]+j) in D:
                A[v][D[(k[0], k[1]+j)]] = -1
            else:
                b[v] += target[k[0]][k[1]+j]

            #fixed_source is g
            #target is f*
            for i in range(3): #three color channel
                #target is uint8, change to int first
                if abs(int(target[k[0]][k[1]][i])-int(target[k[0]+j][k[1]][i]))>abs(int(fixed_source[k[0]][k[1]][i])-int(fixed_source[k[0]+j][k[1]][i])):
                    b[v][i] += int(target[k[0]][k[1]][i])-int(target[k[0]+j][k[1]][i])
                else:
                    b[v][i] += int(fixed_source[k[0]][k[1]][i])-int(fixed_source[k[0]+j][k[1]][i])

                if abs(int(target[k[0]][k[1]][i])-int(target[k[0]][k[1]+j][i]))>abs(int(fixed_source[k[0]][k[1]][i])-int(fixed_source[k[0]][k[1]+j][i])):
                    b[v][i] += int(target[k[0]][k[1]][i])-int(target[k[0]][k[1]+j][i])
                else:
                    b[v][i] += int(fixed_source[k[0]][k[1]][i])-int(fixed_source[k[0]][k[1]+j][i])

    x = cg(A, b)

    for k, v in D.items():
        for i in range(3):
            if x[v][i]>255:
                target[k[0]][k[1]][i] = np.uint8(255)
            elif x[v][i]<0:
                target[k[0]][k[1]][i] = np.uint8(0)
            else:
                target[k[0]][k[1]][i] = np.uint8(round(x[v][i]))

    cv2.imwrite("result_0{0}.jpg".format(pic_index), target)
