# -*- coding: utf-8 -*-
import cv2
import numpy as np

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


for pic_index in range(1, 5):
    mask = cv2.imread("../data/mask_0{0}.jpg".format(pic_index), 0)
    source = cv2.imread("../data/source_0{0}.jpg".format(pic_index))
    target = cv2.imread("../data/target_0{0}.jpg".format(pic_index))

    fixed_source, D = fix_source(source, mask, target.shape, offset[pic_index-1]) #fixed source, same size with target

    for k, v in D.items():
        target[k[0]][k[1]] = fixed_source[k[0]][k[1]]

    cv2.imwrite("result_0{0}.jpg".format(pic_index), target)
