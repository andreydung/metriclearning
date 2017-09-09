from Steerable import Steerable
from skimage.io import imread
from skimage.transform import rescale
import os
import numpy as np
import itertools
import cv2


def gen82Feature(path):
    print(path)
    im = imread(path, as_grey=True)

    ss = Steerable(height=5)
    M, N = im.shape
    coeff = ss.build_scf_pyramid(im)
    coeff_list = [t for sublist in coeff for t in sublist]

    print(len(coeff_list))

    f = []
    # single subband statistics
    for s in coeff_list:
        s = s.real
        shiftx = np.roll(s, 1, axis=0)
        shifty = np.roll(s, 1, axis=1)

        f.append(np.mean(s))
        f.append(np.var(s))
        f.append((shiftx * s).mean() / s.var())
        f.append((shifty * s).mean() / s.var())

    # correlation statistics
    # across orientations
    for orients in coeff[1:-1]:
        for (s1, s2) in list(itertools.combinations(orients, 2)):
            f.append((s1.real * s2.real).mean())

    for orient in range(len(coeff[1])):
        for height in range(len(coeff) - 3):
            s1 = coeff[height + 1][orient].real
            s2 = coeff[height + 2][orient].real

            s1 = rescale(s1, 0.5)
            f.append((s1 * s2).mean() / np.sqrt(s1.var()) / np.sqrt(s2.var()))
    
    # im = cv2.imread(path)
    # hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    # hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # hist = cv2.resize(hist, (0,0), fx = 1/16, fy = 1/16)
    # hist = np.ravel(hist)

    # f += list(hist)
    return np.array(f)


folder = '/home/andrey/Dropbox/Hacking/Research/VisiProg2/firebase/curetaliasfree/static/CURET/images'
listFile = '/home/andrey/Dropbox/Hacking/Research/VisiProg2/firebase/curetaliasfree/static/CURET/list.txt'


with open(listFile) as f:
    paths = f.readlines()
paths = [p.strip() for p in paths]

feature = []
for p in paths:
    f = gen82Feature(os.path.join(folder, p))
    feature.append(f)

feature = np.array(feature)
print(feature.shape)
np.savetxt('curetaliasfree.csv', feature, delimiter=',')