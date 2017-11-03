from .steerable import Steerable
from skimage.io import imread
from skimage.transform import rescale
import numpy as np
import itertools


L = np.load('stsim_trained/trained.npy')


def distance(path1, path2):
	f1 = gen82Feature(path1)
	f2 = gen82Feature(path2)

	return np.linalg.norm(f1.dot(L.T) - f2.dot(L.T))


def gen82Feature(path):
    im = imread(path, as_grey=True)

    ss = Steerable(height=5)
    M, N = im.shape
    coeff = ss.build_scf_pyramid(im)
    coeff_list = [t for sublist in coeff for t in sublist]


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
    
    return np.array(f)



