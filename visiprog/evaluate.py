from sklearn.neighbors import NearestNeighbors
import numpy as np 
from .data import read_material_label
from sklearn.model_selection import train_test_split
import logging


logger = logging.getLogger(__name__)


def leave_one_sample_out(X_train, Y_train, X_test = None, Y_test=None):
	'''
	given path to a feature for each sample
	calculate the leave one sample out accuracy
	'''

	# one nearest neighbor

	nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X_train)

	if X_test is None and Y_test is None:
		logger.info("Nearest neighbor on same data")
		distances, indices = nbrs.kneighbors(X_train)

		# since test and train are same, first will return duplicate
		prediction = Y_train[indices[:,1]]
		accuracy = np.mean(prediction == Y_train)

	elif not X_test is None and not Y_test is None:
		logger.info("Nearest neighbor on separate test data")
		distances, indices = nbrs.kneighbors(X_test)

		# logger.debug(distances)
		logger.debug(indices)

		prediction = Y_train[indices[:,0]]
		accuracy = np.mean(prediction == Y_test)
	else:
		raise ValueError


	results = {}
	results['accuracy'] = accuracy
	results['prediction'] = prediction

	return results
