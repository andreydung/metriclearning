from __future__ import division
import numpy as np
import json
from sklearn.cluster import DBSCAN
import os
from sklearn import manifold
import pandas as pd
from scipy.spatial import Voronoi
import logging
import networkx as nx


logger = logging.getLogger(__name__)


def polar_to_euclidean(theta, phi):
    '''
    theta is polar angle (angle from z axis)
    phi is azimuth angle (angle from x axis to projection)
    
    '''
    X = np.sin(theta) * np.cos(phi)
    Y = np.sin(theta) * np.sin(phi)
    Z = np.cos(theta)
    return (X, Y, Z)


def angle_between(v1, v2):
    ''' Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    '''

    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def read_raw_feature():
    '''
    raw feature of CURET dataset
    using STSIM-M features
    '''

    raw_feature = np.genfromtxt('visiprog/data/curetaliasfree.csv',delimiter=",")
    return raw_feature


def read_img_list():
    '''
    read list of images in CURET
    as a pandas dataframe
    use index column to access (not location index)
    '''
    
    folder = 'visiprog/data/curetaliasfree/images'
    listFile = 'visiprog/data/curetaliasfree/list.txt'
    with open(listFile) as f:
        paths = f.readlines()
    paths = [os.path.join(folder, p.strip()) for p in paths]
    return paths


def read_viewing_conditions():
    '''
    read the viewing condition index
    also calculate the illum and view in Cartesian coordinates
    '''
    df = pd.read_csv('visiprog/data/viewing.csv', index_col=0)

    cartesian = []

    for i in range(df.shape[0]):
        illum_x, illum_y, illum_z = polar_to_euclidean(df['illum_theta'].iloc[i],df['illum_phi'].iloc[i])
        view_x, view_y, view_z = polar_to_euclidean(df['view_theta'].iloc[i],df['view_phi'].iloc[i])

        cartesian.append((illum_x, illum_y, illum_z, view_x, view_y, view_z))

    cartesian = np.array(cartesian)

    df['illum_x'] = cartesian[:,0]
    df['illum_y'] = cartesian[:,1]
    df['illum_z'] = cartesian[:,2]
    df['view_x'] = cartesian[:,3]
    df['view_y'] = cartesian[:,4]
    df['view_z'] = cartesian[:,5]
    
    return df


def viewing_condition_index(img_name):
    '''
    return viewing index based on image name

    deal with m as well (put for fixed alias images)

    '''

    viewing_index = int(os.path.basename(img_name)[3:6])

    return viewing_index


def get_VSP_label(groups, N):
    '''
    read VSP label, to be used for metric training
    False elements mean that it's not labeled by VSP
    '''

    label_visiprog = np.zeros((N,))

    count = 0
    for g in groups:
        label = False
        
        # check whether any entry in the group has been labelled yet
        # if yes, then merge with that existing label
        for i in g:
            if label_visiprog[i] != 0:
                label = label_visiprog[i]
                break
        
        # if no, then create a new label
        if label == False:
            count += 1
            label = count
            
        for i in g:
            label_visiprog[i] = label

    return label_visiprog


def read_VSP_group(pappas_only=True, sorted_by_material=True):
    '''
    Read ViSIProg groups
    '''

    with open('visiprog/data/variedvisiprog-export.json') as f:
        data = json.load(f)
        trials = data['CURETALIASFREE']['trial']
        groups = data['CURETALIASFREE']['group']

    keys = list(trials.keys())
    keys.sort()

    groups = []
    for key in keys:
        entry = trials[key]

        if entry['complete'] == 'true':
            if pappas_only:
                if entry['user'] != 'pappas@eecs.northwestern.edu':
                    continue
            
            group = entry['group']
            gs = [int(g) for g in group.strip().split(',')]

            groups.append(gs)

    sorted_material = None

    if sorted_by_material:
        materials = []
        list_img = read_img_list()

        for g in groups:
            m = [int(os.path.basename(list_img[i])[:2]) for i in g]

            if len(np.unique(np.array(m))) != 1:
                material = -1
            else:
                material = m[0]

            materials.append(material)

        # sort groups based on material
        groups = [g for _, g in sorted(zip(materials, groups))]
        sorted_material = sorted(materials)

    return groups, sorted_material


def get_old_visiprog_coverage():

    filepath = 'visiprog/data/visiprogv1-export.json'

    with open(filepath) as f:
        data = json.load(f)
        trials = data['CURETALIASFREE']['trial']

        counter = {}

        ratio = []
        coverage = 0

        N_TOTAL = 5245

        for k in sorted(trials.keys()):
            v = trials[k]

            if v['complete'] == 'true':

                ids = [int(s) for s in v['group'].strip().split(',')]
                
                for i in ids:
                    if not i in counter:
                        counter[i] = True
                        coverage += 1

                ratio.append(100 * coverage/N_TOTAL)

    return ratio


def get_varied_visiprog_coverage():

    filepath = 'visiprog/data/variedvisiprog-export.json'

    with open(filepath) as f:
        data = json.load(f)
        trials = data['CURETALIASFREE']['trial']

        counter = {}

        ratio = []
        coverage = 0

        N_TOTAL = 5245


        for k in sorted(trials.keys()):
            v = trials[k]

            if v['complete'] == 'true':

                ids = [int(s) for s in v['group'].strip().split(',')]
                
                for i in ids:
                    if not i in counter:
                        counter[i] = True
                        coverage += 1

                ratio.append(100 * coverage/N_TOTAL)

    return ratio


def illum_spatial_adjacent_graph():

    df_viewing = read_viewing_conditions()
    illum_points = df_viewing[['illum_theta', 'illum_phi']].as_matrix()
    illum_vor = Voronoi(illum_points)
    tmp = illum_vor.ridge_points

    exclusion = [[19,1],[19,10], [21,19], [39,41], [30,19],[80,30],[135,94],[53,99],[0,1],[3,1], [54,45], [1,145]]

    illum_edges = []
    for edge in tmp:

        taken = True
        for e in exclusion:
            if set(e) == set(edge):
                taken = False
                break
    
        if taken:
            illum_edges.append(edge)

    illum_edges.extend([[99,49],[94,37],[39,81],[30,22],[53,50], [99,42],[94,81]])

    G_illum = nx.Graph()
    G_illum.add_nodes_from(df_viewing.index.values)

    for e in illum_edges:
        G_illum.add_edge(df_viewing.index.values[e[0]], df_viewing.index.values[e[1]])

    # deal with duplicate items
    duplicates = df_viewing.groupby(['illum_theta', 'illum_phi'])

    for v, d in duplicates.groups.items():
        for i in range(1,len(d)):
            G_illum.add_edge(d[i-1], d[i])

    return G_illum


def viewing_spatial_adjacent_graph():
    df_viewing = read_viewing_conditions()
    viewing_points = df_viewing[['view_theta', 'view_phi']].as_matrix()
    viewing_vor = Voronoi(viewing_points)
    tmp = viewing_vor.ridge_points

    exclusion = [[1,19],[1,167],[1,140],[1, 87],[1,94],[1,39],[10,39],[10,46],[46,56],[51,56],[51,64],[54,32],[10,135],[130,56],[96,64],[89,72],[19,5],[19,3],[19,0]]

    viewing_edges = []
    for edge in tmp:
        # if np.abs(df_viewing['illum_y'].iloc[edge[0]]) < 1e-3 and \
        #     np.abs(df_viewing['illum_y'].iloc[edge[1]]) < 1e-3:

        taken = True
        for e in exclusion:
            if set(e) == set(edge):
                taken = False
                break
    
        if taken:
            viewing_edges.append(edge)

    viewing_edges.extend([[142,144],[96,93], [51,98],[135,100], [39,190],[167,134]])    


    G_viewing = nx.Graph()
    G_viewing.add_nodes_from(df_viewing.index.values)

    for e in viewing_edges:
        G_viewing.add_edge(df_viewing.index.values[e[0]], df_viewing.index.values[e[1]])

    # deal with duplicate items
    duplicates = df_viewing.groupby(['view_theta', 'view_phi'])

    for v, d in duplicates.groups.items():
        for i in range(1,len(d)):
            G_viewing.add_edge(d[i-1], d[i])

    return G_viewing


def read_material_label(merge_zoom=True):
    '''
    Read material label
    Start from 0 (needed for metric learning)
    '''

    label = np.genfromtxt('visiprog/data/label.csv', delimiter = ',').astype(int)
    
    if merge_zoom:
        label[label==29] = 2
        label[label==30] = 11
        label[label==31] = 12
        label[label==32] = 14

        for i in range(33, 62):
            label[label == i] = i - 4

    label = label - 1
    assert np.min(label) == 0

    return label


def count_matrix(groups, N):
    """
    Calculate pairwise matrix from 
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


def distToSimilarity(D):
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

    logger.info("After removing non zero entries, N is %d" % Nred)

    Sm = np.multiply(Dred, 1 - np.eye(Nred))
    Sm = Sm/(np.max(Dred) + 1) + np.eye(Nred)           # diagonal is 1, side is calibrated to 1

    return Sm, nonZeroIndex



def spectralClusteringDBSCAN(D, threshold, eps, min_samples=9, manual_edit=True):
    """ spectral Clustering on a distance matrix
     return the clusters as a list of list of indexes
    """
    # check that D is symmetric
    assert np.allclose(D.T, D)

    # thresholding
    D[D < threshold] = 0

    N = D.shape[0]
    Sm, nonZeroIndex = distToSimilarity(D)

    # Laplacian matrix
    Dmat = np.diag(np.sum(Sm, axis = 0))    # degree matrix
    Lmat = Dmat - Sm                        # laplacian L = D - A

    # eigenvalue and eigenvector decomposition
    eigenValues, eigenVectors = np.linalg.eig(Lmat)

    eigenValues = np.abs(eigenValues)
    eigenVectors = np.abs(eigenVectors)

    # sort so that smallest eigenvalues go first
    idx = eigenValues.argsort()
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]


    N_disconnected_clusters = np.sum(eigenValues < eps)
    logger.info("Number of disconnected elements is %d\n" % N_disconnected_clusters)

    X = eigenVectors[:,:N_disconnected_clusters] 
    db = DBSCAN(eps, min_samples).fit(X)

    labels = db.labels_

    clusters = []
    for value in set(labels):
        c = list(np.where(labels == value)[0])
        clusters.append(c)

    # since D is truncated, need to convert index to original one
    clustersBeforeTruncate = []
    for c in clusters:
        tmp = [nonZeroIndex[i] for i in c]
        clustersBeforeTruncate.append(tmp)

    return clustersBeforeTruncate


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

