from sklearn.neighbors import NearestNeighbors
import numpy as np


def leave_one_sample_out(X):
	'''
	given path to a feature for each sample
	calculate the leave one sample out accuracy
	'''
	label = np.genfromtxt('/Users/andrey/Dropbox/Hacking/Research/VisiProg2/analysis/thesis/visiprog/data/label.csv', delimiter = ',')

	# one nearest neighbor
	nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
	distances, indices = nbrs.kneighbors(X)

	prediction = label[indices[:,1]]

	accuracy = np.mean(prediction == label)

	results = {}
	results['accuracy'] = accuracy
	results['prediction'] = prediction

	return results