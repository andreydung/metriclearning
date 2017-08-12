import numpy as np
import json
from sklearn import cluster
import matplotlib.pyplot as plt
import os
import scipy
from sklearn.cluster import DBSCAN
from sklearn import manifold
from sklearn.metrics import euclidean_distances


def readData():
    N = 5245
    N_group = 9

    with open('../backupData/variedvisiprog-export.json') as f:
        data = json.load(f)
        trials = data['CURETALIASFREE']['trial']
        groups = data['CURETALIASFREE']['group']

    keys = list(trials.keys())
    keys.sort()

    groups = []
    for key in keys:
        entry = trials[key]

        if entry['complete'] == 'true' and entry['user'] == 'pappas@eecs.northwestern.edu':
            group = entry['group']
            gs = [int(g) for g in group.strip().split(',')]

            groups.append(gs)

    return groups, N


def count_matrix(groups, N):
    # fill in the entries of distance matrix
    D = np.zeros((N,N))
    N_group = len(groups[0])
    for g in groups:
        assert len(g) == N_group
        for i in range(N_group):
            D[g[i]][g[i]] += 1
            for j in range(i):
                D[g[i], g[j]] += 1
                D[g[j], g[i]] += 1

    return D

def dis_to_similarity(D, large_value = 15):
    """
    Convert distance matrix to similarity matrix
    Remove all zero rows
    Identical image --> 0
    Mostly labelled --> 1
    0               --> large number
    etc
    """
    N = D.shape[0]

    # remove entries with all zeroes
    # means that it is not covered by ViSiProg
    marginal_sum = np.sum(np.multiply(D, 1 - np.eye(N)), axis = 0)
    nonZeroIndex = np.where(marginal_sum > 0)[0]
    Dred = D[:, nonZeroIndex][nonZeroIndex,:]

    # convert to similarity matrix
    Nred = Dred.shape[0]
    print("After removing non zero entries, N is %d" % Nred)

    # set the diagonal to be 0
    S = np.multiply(Dred, 1 - np.eye(Nred))
    S = np.max(S) + 1 - S

    np.fill_diagonal(S, 0)
    max_value = np.max(S)
    S[S == max_value] = large_value

    return S, nonZeroIndex


def dis_to_similarity_jana(D, threshold):
    """
    Convert distance matrix to similarity matrix
    Remove all zero rows
    """
    N = D.shape[0]

    # remove entries with all zeroes
    # means that it is not covered by ViSiProg
    marginal_sum = np.sum(np.multiply(D, 1 - np.eye(N)), axis = 0)
    nonZeroIndex = np.where(marginal_sum > 0)[0]
    Dred = D[:, nonZeroIndex][nonZeroIndex,:]

    # convert to similarity matrix
    Nred = Dred.shape[0]

    print("After removing non zero entries, N is %d" % Nred)


    Sm = np.multiply(Dred, 1 - np.eye(Nred))
    Sm = Sm/(np.max(Dred) + 1) + np.eye(Nred)           # diagonal is 1, side is calibrated to 1
    # Dmat = np.diag(np.sum(Sm, axis = 0))    # degree matrix
    # Lmat = Dmat - Sm

    return 1 - Sm, nonZeroIndex

def dis_to_similarity_replace(D, threshold, replace=0):
    """
    Convert distance matrix to similarity matrix
    Remove all zero rows
    """
    N = D.shape[0]

    # remove entries with all zeroes
    # means that it is not covered by ViSiProg
    marginal_sum = np.sum(np.multiply(D, 1 - np.eye(N)), axis = 0)
    nonZeroIndex = np.where(marginal_sum > 0)[0]
    Dred = D[:, nonZeroIndex][nonZeroIndex,:]

    # convert to similarity matrix
    Nred = Dred.shape[0]

    print("After removing non zero entries, N is %d" % Nred)

    Sm = np.multiply(Dred, 1 - np.eye(Nred))
    Sm = Sm/(np.max(Dred) + 1) + np.eye(Nred)           # diagonal is 1, side is calibrated to 1

    
    return Sm, nonZeroIndex

def dis_to_similarity_inverse(D, threshold, replace=10):
    """
    Convert distance matrix to similarity matrix
    Remove all zero rows
    """
    N = D.shape[0]

    # remove entries with all zeroes
    # means that it is not covered by ViSiProg
    marginal_sum = np.sum(np.multiply(D, 1 - np.eye(N)), axis = 0)
    nonZeroIndex = np.where(marginal_sum > 0)[0]
    Dred = D[:, nonZeroIndex][nonZeroIndex,:]

    # convert to similarity matrix
    Nred = Dred.shape[0]

    print("After removing non zero entries, N is %d" % Nred)

    Sm = np.multiply(Dred, 1 - np.eye(Nred))
    Sm = Sm/(np.max(Dred) + 1) + np.eye(Nred)           # diagonal is 1, side is calibrated to 1

    return 1 + np.log(Sm + 0.1), nonZeroIndex



if __name__ == "__main__":

    # D = count_matrix([[0,1],[1,2]], 4)
    # S, nonZeroIndex = dis_to_similarity(D)

    seed = np.random.RandomState(seed=3)
    n_samples = 10
    X_true = seed.randint(0, 20, 2 * n_samples).astype(np.float)
    X_true = X_true.reshape((n_samples, 2))
    S = euclidean_distances(X_true)

    # groups, N = readData()
    # D = count_matrix(groups, N)
    # S, nonZeroIndex = dis_to_similarity(D, large_value=100)
    # print(S)

    # np.savetxt('Similarity.csv', S, fmt='%d')

    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=100, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=8)

    pos = mds.fit(S).embedding_
    plt.scatter(pos[:, 0], pos[:, 1],lw=0, label='MDS')

    plt.show()

