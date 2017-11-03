from stsim_trained.metric import distance
from skimage.io import imread
import numpy as np


def calculate_distance(path1, path2):

	patch_size = 64
	grid_size = 64

	im1 = imread(path1, as_grey=True)
	im2 = imread(path2, as_grey=True)

	M, N = im1.shape

	dists = []

	for i in range(0, M - patch_size + 1, grid_size):
		# print(i)
		for j in range(0, N - patch_size + 1, grid_size):

			patch1 = im1[i:i+patch_size, j:j+patch_size]
			patch2 = im2[i:i+patch_size, j:j+patch_size]

			dist = distance(patch1, patch2)

			dists.append(dist)

	dists = np.array(dists)
	return np.mean(dists)


print("Warped")
print(calculate_distance('distorted/original.png','distorted/warped.png'))

print("Very Warped")
print(calculate_distance('distorted/original.png','distorted/warped_verydistorted.png'))


print("Rotated")
print(calculate_distance('distorted/original.png','distorted/rotated.png'))

print("Very Rotated")
print(calculate_distance('distorted/original.png','distorted/rotated_verydistorted.png'))

print("Shifted")
print(calculate_distance('distorted/original.png','distorted/shifted.png'))

print("Very Shifted")
print(calculate_distance('distorted/original.png','distorted/shifted_verydistorted.png'))
