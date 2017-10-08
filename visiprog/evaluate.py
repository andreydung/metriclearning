from sklearn.neighbors import NearestNeighbors
import numpy as np 
from .data import read_material_label
from sklearn.model_selection import train_test_split


def leave_one_sample_out(X, label):
	'''
	given path to a feature for each sample
	calculate the leave one sample out accuracy
	'''

	# one nearest neighbor
	nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
	distances, indices = nbrs.kneighbors(X)

	prediction = label[indices[:,1]]

	accuracy = np.mean(prediction == label)

	results = {}
	results['accuracy'] = accuracy
	results['prediction'] = prediction

	return results


# def split_material_validate(X, Y, ratio=0.5):

# 	indices = np.arange(X.shape[0])

# 	label_material = read_material_label()

# 	X_train, X_test, Y_train, Y_test = \
# 		train_test_split(X, Y, indices, test_size=ratio, random_state=42, stratify=Y)


# 	return X_train, X_test, Y_train, Y_test
