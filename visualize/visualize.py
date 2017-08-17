import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import numpy as np


def scatter_img(X, imgPath, outPath):

	assert X.shape[0] <= len(imgPath)

	X = X - X.min()
	X = X/X.max() * 6000
	X = X.astype(int)

	out = np.zeros((6500, 7000, 3)).astype(np.int)

	N = X.shape[0]

	im = imread(imgPath[0])
	Mimg = im.shape[0]
	Nimg = im.shape[1]

	for i in range(N):
		im = imread(imgPath[i])
		xpos = X[i][0]
		ypos = X[i][1]

		out[xpos:xpos + Mimg,ypos:ypos + Nimg,:] = im

	imsave(outPath, out)



def scatter_plot(X, label):
    ax = plt.gca()
    return ax.scatter(X[:, 0], X[:, 1],lw=0, label=label)
