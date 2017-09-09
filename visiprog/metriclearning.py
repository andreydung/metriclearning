from metric_learn import LSML, Covariance, ITML, LFDA
import numpy as np
import json
import random
from sklearn.datasets import load_iris
from scipy.linalg import cholesky
import logging


def train_covariance(X):

	model = Covariance()
	model.fit(X)

	return model.transform(X), model.metric()


def fisher_discriminant(X, Y):

	model = LFDA()
	model.fit(X, Y)


	return model.transform(X), model.metric()
