import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage import img_as_float
import numpy as np


def scatter_img_networkx(pos, imgPath, outPath, size=6000):

	N = len(pos)
	X = np.zeros((N, 2))

	for i in range(N):
		X[i][0] = pos[i][0]
		X[i][1] = pos[i][1]

	scatter_img(X, imgPath, outPath, size)


def scatter_img(X, imgPath, outPath, size=6000):

	assert X.shape[0] <= len(imgPath)

	X = X - X.min()
	X = X/X.max() * size *0.9
	X = X.astype(int)

	out = np.zeros((size, size, 3))

	N = X.shape[0]

	im = imread(imgPath[0])
	Mimg = im.shape[0]
	Nimg = im.shape[1]

	for i in range(N):
		im = imread(imgPath[i])
		xpos = X[i][0]
		ypos = X[i][1]

		out[xpos:xpos + Mimg,ypos:ypos + Nimg,:] = img_as_float(im)

	imsave(outPath, out)



def scatter_plot(X, label):
    ax = plt.gca()
    return ax.scatter(X[:, 0], X[:, 1],lw=0, label=label)
