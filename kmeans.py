#! /usr/bin/env python3
# cluster.py
import os, sys
import sklearn.metrics as skm

from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fowlkes_mallows_score
#from sklearn.metrics import matthews_corrcoef
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def compute_centroids(data, classes):
    # collapse data columns into MEAN matching on classes
    # mean of coords gives centroid
    centroid_list = data.groupby(by = classes).mean()

    # hopefully centroid_list is now a list of length( unique(classes))
    # where each row is a centroid and the columns are the centroid coords

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
    
def visualization(data, corr_mat, cluster_labels):

    ## Full-Dimension Visualizations
    # similarity heatmap
    #data = sort_data(by = classes) ??

    #sim_mat = similarity_matrix(data)
    #heat = sns.heatmap(corr_mat)
    #heat.show()

    ## Reduced Dimension Visualizations
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)

    pca_df = pd.DataFrame(data = pca_data, columns = ['PC1', 'PC2'])
 
    print(pca_data)
    print(pca_data.shape)

    # Visualize
    #plt.plot(pca_df['PC1'], pca_df['PC2'], color = cluster_labels)
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
def method_evaluation(data, target = 'sex', optimization_metric = 'Silhuoette'):

    full_data = data.copy() # we need full data (no drop target) for the mode-voting

    target_data = data.loc[:, target]
    target_names = [str(x) for x in target_data.unique().tolist()] # i still dont know what this is

    true_labs = target_data

    data = data.drop(columns=target)

    for k in range(1, 15):

        # Kmeans
        kmeans = KMeans(n_clusters = k).fit(data) #random state = 0 ?
        #print(kmeans.labels_)
        kmeans_clus_labels = kmeans.labels_
        
        kmeans_pred_labs = clusters_to_labels_voting(full_data, kmeans_clus_labels)

        kmeans_k_score.append(evaluate(true_labs, kmeans_pred_labs, data, kmeans.cluster_centers_, k)[0]) # [0] because only taking the list of scores because im not gonna mess w matrix
    
    print('K-Means Clustering Scores per K: ')
    print(kmeans_k_score) # print the table of scores

    method_names = ['SSE', 'Adj Rand', 'Norm Mut Info', 'Adj Mut Info', 'Homog', 'Completeness', 'V-Measure']

    return pd.DataFrame(kmeans_k_score, columns = method_names)

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
    print('preprocessing')
    full_data_names = full_data.columns
    mms = preprocessing.MinMaxScaler()
    mms_full_data = mms.fit_transform(full_data)
    mms_full_data = pd.DataFrame(mms_full_data, columns=full_data_names)

    # Separate out the data for various experiments here
    mms_data = full_data.drop(columns=target)
    data_names = mms_data.columns
    mms_data = mms.fit_transform(mms_data)
    mms_data = pd.DataFrame(mms_data, columns=data_names)

    # Use Method_Evaluation to find optimal K (currently according to SILHUOETTE, but could be any metric)
    #kmeans_scores, agglom_scores, dbscan_scores, spectral_scores = method_evaluation(data, 'sex') # could make argument for which Metric u want optimal K for

    # Optimized K-means
    # can find best score here by looking at matrix
    #    plt.plot(x = kmeans_scores['k'], y = kmeans_scores['silhuoette'])
    #    plt.show()

    # Run the experiments
    print('run the experiments')
    k = 10

    # BTW kmeans gives you kmeans.inertia_, which is SSE

    # With target experiments
    print('with target')
    centroids = []
    kmeans_wt = KMeans(n_clusters = k).fit(mms_data)
    centroids = kmeans_wt.inertia_
    print(kmeans_wt)
    pred_labs_wt = clusters_to_labels_voting(mms_data, kmeans_wt.labels_, target_data, target)
    print('goit pred labs wt')
    #def evaluate_internal(true_labs, cluster_labs, pred_labs, data, centroids, prox_mat):
    internal_scores_list = evaluate_internal(true_labs = target_data, 
                                                           cluster_labs = kmeans_wt.labels_, 
                                                           pred_labs = pred_labs_wt, 
                                                           data = mms_data, 
                                                           centroids = centroids,
                                                           prox_mat = prox_mat)
    print(internal_scores_list)

    # Without target experiments
    print('without target')
    kmeans_wot = KMeans(n_clusters = k).fit(mms_full_data)
    centroids = kmeans_wot.inertia_
    pred_labs_wot = clusters_to_labels_voting(mms_full_data, kmeans_wot.labels_, target_data, target)
    external_scores_list, cont_mat = evaluate_external(target_data, pred_labs_wot)
    print(external_scores_list)
    print(cont_mat)
    #visualization(scaled_data_train_wot, cont_mat, target)
    sys.exit()



# https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
# noel read this ^^ really good code that overall i think we should look like
# It's pretty good. I like the print statements, but they are only doing it for 1 k value. I think in general, we're on a good track.
