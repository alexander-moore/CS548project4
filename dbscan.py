#! /usr/bin/env python3
# cluster.py
import os, sys
import sklearn.metrics as skm

from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fowlkes_mallows_score
#from sklearn.metrics import matthews_corrcoef
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def compute_centroids(data, classes):
    # collapse data columns into MEAN matching on classes
    # mean of coords gives centroid
    centroid_list = data.groupby(by = classes).mean()

    return centroid_list

# Evaluate internal/relative indicies of a clustering given inputs
def evaluate_internal(true_labs, cluster_labs, pred_labs, data, centroids, prox_mat):
    #SSE = need_SSE_func(centroids, pred_labs) # centers are given by the methods
    adj_rand = skm.adjusted_rand_score(true_labs, pred_labs)
    norm_info = skm.normalized_mutual_info_score(true_labs, pred_labs)
    adj_info = skm.adjusted_mutual_info_score(true_labs, pred_labs)
    silhuoette = skm.silhouette_score(prox_mat, cluster_labs) # need to verify arguments on this
    
    # Correlation Cluster Validity
    match_matrix = np.zeros((len(data.index), len(data.index)))
    for i in range(0, len(data.index)):
        row_lab = cluster_labs[i]
        for j in range(0, len(data.index)):
            if row_lab == cluster_labs[j]:
                match_matrix[i, j] = 1
            else:
                match_matrix[i, j] = 0

    # correlation of elements of prox_mat and match_mat, ideally close to -1:
    clus_cor = np.corrcoef(prox_mat.to_numpy().flatten(), match_matrix.flatten())

    # Corr matrix of 2 matrices??? Leave for the time being
    #clus_cor_matrix = np.corrcoef(prox_mat, match_matrix)
    #return [adj_rand, norm_info, adj_info, silhuoette, clus_cor[0,1]] #, clus_cor_matrix

    # NOTE: returning the correlation between 2 LOOONG arrays, also returning the corr matrix between the two matrices???? Let if for now I guess
    return [adj_rand, norm_info, adj_info, silhuoette, clus_cor[0,1]]
    #return [adj_rand, norm_info, adj_info, silhuoette, 1]

# Evaluate external indicies of a clustering given true and predicted labels
def evaluate_external(true_labs, pred_labs):
    homog = skm.homogeneity_score(true_labs, pred_labs)
    complete = skm.completeness_score(true_labs, pred_labs)
    v_measure = skm.v_measure_score(true_labs, pred_labs)
    cont_mat = skm.cluster.contingency_matrix(true_labs, pred_labs)
    return [homog, complete, v_measure], cont_mat
    
def visualization(data, similarity, corr_mat, cluster_labels, title):
    print('hi. welcome to visualization:')

    ## Full-Dimension Visualizations
    # similarity heatmap
    #sim_sorted = similarity.sort_data(by = classes)
    #heat = sns.heatmap(sim_sorted)
    #heat.show()

    ## Reduced Dimension Visualizations
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)

    pca_df = pd.DataFrame(data = pca_data, columns = ['PC1', 'PC2'])

    # Visualize
    plt.scatter(pca_df['PC1'], pca_df['PC2'], c = cluster_labels, alpha = .5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.show()

    # nonparametric MDS takes a long time, only run if you have to
    #mds = MDS(n_components = 2, metric = False)
    #mds_data = mds.fit_transform(data)
    #print(mds_data.shape)
    #mds_df = pd.DataFrame(data = mds_data, columns = ['1', '2'])

    #plt.scatter(mds_df['1'], mds_df['2'], c = cluster_labels, alpha = .5)
    #plt.xlabel('MDS dim 1')
    #plt.ylabel('MDS dim 2')
    #plt.title(title)
    #plt.show()





# Turns arbitrary cluster labelling [clus1, clus2, ..] into the same type as our target (ex. 1 male 0 female) by getting the mode target of each cluster
def clusters_to_labels_voting(data, clus_labels, target_labels, target):

    method_pred_labs = np.zeros(len(clus_labels))
    #target_index = data.columns.get_loc(target)
    #print(target_index)

    for clus in set(clus_labels): # for each cluster
        cluster = []
        index_list = []
        for i in range(0, len(clus_labels)):
            if clus_labels[i] == clus:
                index_list.append(i)
                cluster.append(target_labels.iloc[i])
        predicted_target = max(set(cluster), key = cluster.count)
        for index in index_list:
            method_pred_labs[index] = predicted_target

    return method_pred_labs

# method_evaluation should create, display, and select best scoring K across methods
def method_evaluation(full_data, data, prox_mat, target_data, target = 'sex', optimization_metric = 'Silhuoette'):
    scores = np.zeros(9)
    for k in range(2, 50, 3):
        scores_for_k = [k]
        print('k = ' + str(k))
        # With target experiments
        print('DBSCAN with target')
        dbscan_wt = DBSCAN(eps = k/400).fit(full_data)
        centroids = compute_centroids(full_data, dbscan_wt.labels_)
        pred_labs_wt = clusters_to_labels_voting(full_data, dbscan_wt.labels_, target_data, target)
        #def evaluate_internal(true_labs, cluster_labs, pred_labs, data, centroids, prox_mat):
        internal_scores_list = evaluate_internal(true_labs = target_data, 
                                                 cluster_labs = dbscan_wt.labels_, 
                                                 pred_labs = pred_labs_wt, 
                                                 data = full_data, 
                                                 centroids = centroids,
                                                 prox_mat = prox_mat)
        print('[adj_rand, norm_info, adj_info, silhuoette, clus_cor[0,1]]')
        print(internal_scores_list)
        scores_for_k.extend(internal_scores_list)

        # Without target experiments
        print('DBSCAN without target')
        dbscan_wot = DBSCAN(eps = k/400).fit(data)
        centroids = compute_centroids(data, dbscan_wot.labels_)
        pred_labs_wot = clusters_to_labels_voting(data, dbscan_wot.labels_, target_data, target)
        external_scores_list, cont_mat = evaluate_external(target_data, pred_labs_wot)
        print('[homog, complete, v_measure], cont_mat')
        print(external_scores_list)
        print(cont_mat)
        scores_for_k.extend(external_scores_list)
        print(scores_for_k)
        if k == 2:
            scores = np.array(scores_for_k)
        else:
            scores = np.vstack((scores, np.array(scores_for_k)))
    
    print('K-Means Clustering Scores per K: ')
    print(scores) # print the table of scores

    method_names = ['k', 'Adj Rand', 'Norm Mut Info', 'Adj Mut Info', 'Silhuoette', 'Clus_cor', 'Homog', 'Completeness', 'V-Measure']

    return pd.DataFrame(scores, columns = method_names)

def plot_method_eval(scores):
    for i in range(0, scores.shape[1] - 1):
        plt.figure(i)
        plt.plot(scores.loc[:, 'k'], scores.iloc[:, i+1])
        plt.title('{} graphed over varying k'.format(scores.columns[i+1]))
    plt.show()

def plot_method_eval_from_csv(csv):
    scores = pd.read_csv(csv)
    for i in range(0, scores.shape[1] - 1):
        plt.figure(i)
        plt.plot(scores.loc[:, 'k'], scores.iloc[:, i+1])
        plt.title('{} graphed over varying k'.format(scores.columns[i+1]))
    plt.show()

if __name__ == '__main__':
    # Delcare target
    target = 'sex'

    # Load the data here
    print('loading...')
    full_data = pd.read_csv('tiny_cci.csv')
    prox_mat = pd.read_csv('tiny_prox_mat.csv')
    print('loaded')

    target_data = full_data.loc[:, target].copy()

    # transform to pd
    print('Preprocessing')
    full_data_names = full_data.columns
    mms = preprocessing.MinMaxScaler()
    mms_full_data = mms.fit_transform(full_data)
    mms_full_data = pd.DataFrame(mms_full_data, columns=full_data_names)

    # Separate out the data for various experiments here
    mms_data = full_data.drop(columns=target)
    data_names = mms_data.columns
    mms_data = mms.fit_transform(mms_data)
    mms_data = pd.DataFrame(mms_data, columns=data_names)



    # Write directly to experiments:
    # n clusters:
    dbscan = DBSCAN(min_samples = 5).fit(mms_data)
    #print(kmeans.inertia_ )
    print(Counter(dbscan.labels_))
    #visualization(data = mms_data, similarity = 3, corr_mat = 1, cluster_labels = kmeans_exp1.labels_, title = 'Unsupervised K=2')
    #[homog, complete, v_measure], cont_mat
    print(evaluate_external(target_data, clusters_to_labels_voting(mms_full_data, dbscan.labels_, target_data, target)))

    dbscan = DBSCAN(min_samples = 25).fit(mms_data)
    #print(kmeans.inertia_)
    print(Counter(dbscan.labels_))
    #visualization(data = mms_data, similarity = 3, corr_mat = 1, cluster_labels = kmeans.labels_, title = 'Unsupervised K=10')
    print(evaluate_external(target_data, clusters_to_labels_voting(mms_full_data, dbscan.labels_, target_data, target)))

    dbscan = DBSCAN(min_samples = 100).fit(mms_data)
    #print(kmeans.inertia_)
    print(Counter(dbscan.labels_))
    #visualization(data = mms_data, similarity = 3, corr_mat = 1, cluster_labels = kmeans_exp3.labels_, title = 'Unsupervised K=20')
    print(evaluate_external(target_data, clusters_to_labels_voting(mms_full_data, dbscan.labels_, target_data, target)))

    print('exiting')
    sys.exit()
    print('didnt get here')


    # Find a semi optimal k by running a lot and returning a matrix of scores
    scores = method_evaluation(mms_full_data, mms_data, prox_mat, target_data, target = 'sex', optimization_metric = 'Silhuoette')
    scores.to_csv('dbscan_method_eval_scores.csv', index = False)

    print(scores)
    graph_method_eval(scores)
    print('exiting')
    sys.exit()


    # Run the experiments
    print('run the experiments')
    k = 10

    # BTW kmeans gives you kmeans.inertia_, which is SSE

    # With target experiments
    print('with target')
    centroids = []
    dbscan_wt = DBSCAN(n_clusters = k).fit(mms_data)
    centroids = compute_centroids(dbscan_wt.labels_)
    print(dbscan_wt)
    pred_labs_wt = clusters_to_labels_voting(mms_data, dbscan_wt.labels_, target_data, target)
    print('goit pred labs wt')
    #def evaluate_internal(true_labs, cluster_labs, pred_labs, data, centroids, prox_mat):
    internal_scores_list = evaluate_internal(true_labs = target_data, 
                                                           cluster_labs = dbscan_wt.labels_, 
                                                           pred_labs = pred_labs_wt, 
                                                           data = mms_data, 
                                                           centroids = centroids,
                                                           prox_mat = prox_mat)
    print('[adj_rand, norm_info, adj_info, silhuoette, clus_cor[0,1]]')
    print(internal_scores_list)

    visualization(data = mms_data, similarity = prox_mat, corr_mat = 1, cluster_labels = dbscan_wt.labels_, title = 'Supervised K-Means (10)')

    k = 2
    print('k = 2:')

    centroids = []
    dbscan_wt = DBSCAN(n_clusters = k).fit(mms_data)
    centroids = compute_centroids(mms_data, dbscan_wt.labels_)
    print(dbscan_wt)
    pred_labs_wt = clusters_to_labels_voting(mms_data, dbscan_wt.labels_, target_data, target)
    print('goit pred labs wt')
    #def evaluate_internal(true_labs, cluster_labs, pred_labs, data, centroids, prox_mat):
    internal_scores_list = evaluate_internal(true_labs = target_data, 
                                                           cluster_labs = dbscan_wt.labels_, 
                                                           pred_labs = pred_labs_wt, 
                                                           data = mms_data, 
                                                           centroids = centroids,
                                                           prox_mat = prox_mat)
    print('[adj_rand, norm_info, adj_info, silhuoette, clus_cor[0,1]]')
    print(internal_scores_list)

    visualization(data = mms_data, similarity = prox_mat, corr_mat = 1, cluster_labels = dbscan_wt.labels_, title = 'Supervised K-Means (2)')


    # Without target experiments
    print('without target')
    dbscan_wot = DBSCAN(n_clusters = k).fit(mms_full_data)
    centroids = compute_centroids(mms_full_data, dbscan_wot.labels_)
    pred_labs_wot = clusters_to_labels_voting(mms_full_data, dbscan_wot.labels_, target_data, target)
    external_scores_list, cont_mat = evaluate_external(target_data, pred_labs_wot)
    print('[homog, complete, v_measure], cont_mat')
    print(external_scores_list)
    print(cont_mat)

# https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
# noel read this ^^ really good code that overall i think we should look like
# It's pretty good. I like the print statements, but they are only doing it for 1 k value. I think in general, we're on a good track.
