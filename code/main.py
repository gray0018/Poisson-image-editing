import cv2
import numpy as np

def fix_source(source, mask, shape, offset):
    mydict = {}
    counter = 0

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j]>127:
                mydict[(i+offset[0],j+offset[1])] = counter
                counter += 1
    fixed_source = np.zeros(shape, dtype=int) #注意一定是uint8 不然最后imshow显示不出来#是int 不然后面*4会爆掉
    fixed_source[offset[0]:source.shape[0]+offset[0], offset[1]:source.shape[1]+offset[1],:]=source

    return fixed_source, mydict

offset = [[210, 10]]

mask = cv2.imread("data/mask_01.jpg", 0)
source = cv2.imread("data/source_01.jpg")
target = cv2.imread("data/target_01.jpg")

fixed_source, D = fix_source(source, mask, target.shape, offset[0]) #fixed source, same size with target

A = np.zeros((len(D),len(D)), dtype=int)
b = np.zeros((len(D),3), dtype=int)

for k, v in D.items():
    A[v][v] = 4
    b[v] += 4*fixed_source[k[0]][k[1]] - fixed_source[k[0]+1][k[1]] - b[v] - fixed_source[k[0]-1][k[1]] - fixed_source[k[0]][k[1]+1] - fixed_source[k[0]][k[1]-1]

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
        A[v][D[(k[0], k[1]-1)]]
    else:
        b[v] += target[k[0]][k[1]-1]

x = np.linalg.lstsq(A, b, rcond=None)[0]


for k, v in D.items():
    target[k[0]][k[1]][0] = np.uint8(round(x[v][0])%256)
    target[k[0]][k[1]][1] = np.uint8(round(x[v][1])%256)
    target[k[0]][k[1]][2] = np.uint8(round(x[v][2])%256)

# cv2.imshow("source01", fixed_source)
cv2.imshow("target01", target)
cv2.waitKey()
cv2.destroyAllWindows()
