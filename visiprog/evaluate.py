from sklearn.neighbors import NearestNeighbors
import numpy as np


def leave_one_sample_out(X):
	'''
	given path to a feature for each sample
	calculate the leave one sample out accuracy
	'''
	label = np.genfromtxt('/Users/andrey/Dropbox/Hacking/Research/VisiProg2/analysis/thesis/visiprog/data/label.csv', delimiter = ',')

	nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)

	distances, indices = nbrs.kneighbors(X)
	prediction = label[indices[:,1]]

	return (np.mean(prediction == label))
