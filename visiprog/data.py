import numpy as np
import json
from sklearn import cluster
import os
import scipy
from sklearn import manifold


def get_raw_feature():
    raw_feature = np.genfromtxt('visiprog/data/curetaliasfree.csv',delimiter=",")
    return raw_feature


def read_img_list():
    folder = 'visiprog/data/curetaliasfree/images'
    listFile = 'visiprog/data/curetaliasfree/list.txt'
    with open(listFile) as f:
        paths = f.readlines()
    paths = [os.path.join(folder, p.strip()) for p in paths]
    return paths


def read_VSP_label():
    N = 5245
    N_group = 9

    with open('visiprog/data/variedvisiprog-export.json') as f:
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


def read_material_label():
    label = np.genfromtxt('visiprog/data/label.csv', delimiter = ',').astype(int)
    label = label - 1
    return label


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
    0               --> missing_value
    etc
    """
    S_nonzero, nonZeroIndex = remove_zero_rows(S)

    # convert to similarity matrix
    # set the diagonal to be 0
    np.fill_diagonal(S_nonzero, 0)
    D = np.max(S_nonzero) + 1 - S_nonzero

    np.fill_diagonal(D, 0)
    D[D == np.max(D)] = missing_value

    return D, nonZeroIndex


def remove_zero_rows(S):
    """ 
    remove entries with all zeroes
    means that it is not covered by ViSiProg
    """
    N = S.shape[0]
    marginal_sum = np.sum(np.multiply(S, 1 - np.eye(N)), axis = 0)
    nonZeroIndex = np.where(marginal_sum > 0)[0]
    S_nonzero = S[:, nonZeroIndex][nonZeroIndex,:]

    return S_nonzero, nonZeroIndex


def Jana_method(S):

    S_nonzero, nonZeroIndex = remove_zero_rows(S)
    Adjacent = S_nonzero/(np.max(S_nonzero) + 1)
    np.fill_diagonal(Adjacent, 0)

    return Adjacent, nonZeroIndex



def spectral_clustering(S):
    """
    Convert distance matrix to similarity matrix
    Remove all zero rows
    Identical image --> 0
    Mostly labelled --> 1
    0               --> large number
    etc
    """

    S_nonzero, nonZeroIndex = remove_zero_rows(S)

    # convert to similarity matrix
    # set the diagonal to be 0
    Adjacent = S_nonzero/(np.max(S_nonzero) + 1)
    np.fill_diagonal(Adjacent, 0)


    print("Adjacent")
    print(Adjacent)

    # Laplacian matrix
    Degree = np.diag(np.sum(Adjacent, axis=0))  # degree matrix
    Laplacian = Degree - Adjacent

    print("Laplacian")
    print(Laplacian)

    eigenValues, eigenVectors = np.linalg.eig(Laplacian)

    eigenValues = np.abs(eigenValues)
    eigenVectors = np.abs(eigenVectors)
    
    idx = eigenValues.argsort()
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]


    # take first ith eigenvectors
    X = eigenVectors[:,:10] 

    return X, nonZeroIndex



def MDS(D):
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=100, eps=1e-1, random_state=seed,
                   dissimilarity="precomputed", n_jobs=8)

    X = mds.fit(D).embedding_
    return X


def SMACOF(D, num_iter, eps):
    X, stress = manifold.smacof(D, metric=False,n_components=2,verbose=2,\
        max_iter=num_iter, eps=eps, n_jobs=8)

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

