import numpy as np
import json
from sklearn import cluster
import os
import scipy
from sklearn import manifold


def readImgList():
    folder = '/Users/andrey/Dropbox/Hacking/Research/VisiProg2/firebase/curetaliasfree/static/CURET/images'
    listFile = '/Users/andrey/Dropbox/Hacking/Research/VisiProg2/firebase/curetaliasfree/static/CURET/list.txt'
    with open(listFile) as f:
        paths = f.readlines()
    paths = [os.path.join(folder, p.strip()) for p in paths]
    return paths


def readVSPLabel():
    N = 5245
    N_group = 9

    with open('/Users/andrey/Dropbox/Hacking/Research/VisiProg2/backupData/variedvisiprog-export.json') as f:
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
    """
    Return pairwise matrix
    that has counts of number of images in the same group
    """
    S = np.zeros((N,N))
    N_group = len(groups[0])
    for g in groups:
        assert len(g) == N_group
        for i in range(N_group):
            S[g[i]][g[i]] += 1
            for j in range(i):
                S[g[i], g[j]] += 1
                S[g[j], g[i]] += 1

    return S


def similarity_to_distance(S, missing_value = 0):
    """
    Convert distance matrix to similarity matrix
    Remove all zero rows
    Identical image --> 0
    Mostly labelled --> 1
    0               --> large number
    etc
    """
    N = S.shape[0]

    # remove entries with all zeroes
    # means that it is not covered by ViSiProg
    marginal_sum = np.sum(np.multiply(S, 1 - np.eye(N)), axis = 0)
    nonZeroIndex = np.where(marginal_sum > 0)[0]
    Sred = S[:, nonZeroIndex][nonZeroIndex,:]
    Nred = Sred.shape[0]

    # convert to similarity matrix
    # set the diagonal to be 0
    np.fill_diagonal(S, 0)
    D = np.max(S) + 1 - S

    np.fill_diagonal(D, 0)
    max_value = np.max(D)
    D[D == max_value] = missing_value

    return D, nonZeroIndex


def spectral_clustering(S):
    """
    Convert distance matrix to similarity matrix
    Remove all zero rows
    Identical image --> 0
    Mostly labelled --> 1
    0               --> large number
    etc
    """
    N = S.shape[0]

    # remove entries with all zeroes
    # means that it is not covered by ViSiProg
    marginal_sum = np.sum(np.multiply(S, 1 - np.eye(N)), axis = 0)
    nonZeroIndex = np.where(marginal_sum > 0)[0]
    Sred = S[:, nonZeroIndex][nonZeroIndex,:]
    Nred = Sred.shape[0]

    # convert to similarity matrix
    # set the diagonal to be 0
    np.fill_diagonal(S, 0)
    D = np.max(S) + 1 - S

    np.fill_diagonal(D, 0)
    max_value = np.max(D)
    D[D == max_value] = missing_value

    return D, nonZeroIndex


def MDS(D):
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=100, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=8)

    X = mds.fit(D).embedding_
    return X


def SMACOF(D):
    X, stress = manifold.smacof(D, metric=False,n_components=2,verbose=2, max_iter=3, eps=1e-9, n_jobs=8)

    return X

# if __name__ == "__main__":

#     # unit testing
#     # D = count_matrix([[0,1],[1,2]], 4)
#     # S, nonZeroIndex = dis_to_similarity(D)


#     # testing MDS
#     seed = np.random.RandomState(seed=3)
#     n_samples = 10
#     X_true = seed.randint(0, 20, 2 * n_samples).astype(np.float)
#     X_true = X_true.reshape((n_samples, 2))
#     S = euclidean_distances(X_true)

#     # groups, N = readData()
#     # D = count_matrix(groups, N)
#     # S, nonZeroIndex = dis_to_similarity(D, large_value=100)
#     # print(S)

#     # np.savetxt('Similarity.csv', S, fmt='%d')

#     seed = np.random.RandomState(seed=3)
#     mds = manifold.MDS(n_components=2, max_iter=100, eps=1e-9, random_state=seed,
#                    dissimilarity="precomputed", n_jobs=8)

#     pos = mds.fit(S).embedding_
#     plt.scatter(pos[:, 0], pos[:, 1],lw=0, label='MDS')

#     plt.show()

