from sklearn.neighbors import NearestNeighbors
import numpy as np 
from .data import read_material_label
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
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


# def find_exemplars_kmeans_whole(X, n_clusters):

# 	km = KMeans(n_clusters).fit(X)
# 	closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)

# 	return closest


def find_exemplars(X, clusters_label):
	exemlpars_index = []

	clusters = [np.where(clusters_label == value)[0] for value in np.unique(clusters_label)]

	# finding exemplars within each cluster
	for cluster in clusters:
		X_cluster = X[np.array(cluster)]

		if len(cluster) >=25:
			n_clusters = 4
		else:
			n_clusters = 2


		km = KMeans(n_clusters).fit(X_cluster)
		closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
		exemlpars_index.append(closest)
	
	return np.concatenate(exemlpars_index, axis=0)


def kmean_subclass(X_train, Y_train, X_test, Y_test, n_clusters):
	'''
	instead of 1 nearest neighbor
	use the subclass exemplar
	'''

	logger.info("KMeans clustering")

	logger.info(Y_train)

	codebook = []
	label = []
	for unique_value in np.unique(Y_train):
		# logger.info(unique_value)
		index = np.where(Y_train == unique_value)
		subset = X_train[index]

		# logger.info(subset.shape[0])
		# logger.info(Y_train[index])

		km = KMeans(n_clusters=n_clusters, init='k-means++')
		km.fit(subset)

		codebook.append(km.cluster_centers_)
		label.append(unique_value * np.ones((n_clusters,)))

	codebook = np.vstack(codebook)
	label = np.hstack(label)

	# use the codebook to validate the test set
	logger.info("Running on test set")
	
	nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(codebook)
	distances, indices = nbrs.kneighbors(X_test)
	prediction = label[indices[:,0]]

	accuracy = np.mean(prediction == Y_test)

	res = {}
	res['accuracy'] = accuracy
	res['prediction'] = prediction

	return res
	