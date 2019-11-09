#! /usr/bin/env python3
# cluster.py
import os, sys
import sklearn.metrics as skm

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import pairwise_distances
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
def evaluate_internal(true_labs, cluster_labs, pred_labs, data, centroids):
    #SSE = need_SSE_func(centroids, pred_labs) # centers are given by the methods
    adj_rand = skm.adjusted_rand_score(true_labs, pred_labs)
    norm_info = skm.normalized_mutual_info_score(true_labs, pred_labs)
    adj_info = skm.adjusted_mutual_info_score(true_labs, pred_labs)
    pairwise_dist = skm.pairwise_distances(data.values)
    silhuoette = skm.silhouette_score(pairwise_dist, cluster_labs) # need to verify arguments on this
    
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
    clus_cor = np.corrcoef(pairwise_dist.flatten(), match_matrix.flatten())
    clus_cor_matrix = np.corrcoef(pairwise_dist, match_matrix)
    #return [SSE, adj_rand, norm_info, adj_info, silhuoette], clus_cor
    # NOTE: returning the correlation between 2 LOOONG arrays, also returning the corr matrix between the two matrices???? Let if for now I guess
    return [adj_rand, norm_info, adj_info, silhuoette, clus_cor[0,1]], clus_cor_matrix

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
    heat = sns.heatmap(corr_mat)
    heat.show()

    ## Reduced Dimension Visualizations
    pca = skm.PCA(n_components=2)
    pca_data = pca.fit(data)
    pca_df = pd.DataFrame(pca_data, columns = ['PC_1', 'PC_2'])

    # Visualize
    plt.plot(x = pca_df['PC_1'], y = pca_df['PC_2'], color = classes, legend = classes)
    plt.show()


# Turns arbitrary cluster labelling [clus1, clus2, ..] into the same type as our target (ex. 1 male 0 female) by getting the mode target of each cluster
def clusters_to_labels_voting(data, clus_labels, target_labels, target):

    method_pred_labs = np.zeros(len(clus_labels))
    target_index = data.columns.get_loc(target)

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
        #print(kmeans.predict(data))
        #print(kmeans.cluster_centers_)

        # Agglom
        #agglom = AgglomerativeClustering(n_clusters = k).fit(data)
        #print(agglom.labels_)
        #agglom_clus_labels = agglom.labels_

        # DBSCAN
        # eps is the ball radius around points. remember our space is in high dimension, but scaled to 0,1. so this is fucked
        core_samples, labels = DBSCAN(eps = .05, min_samples = k).fit(data) #takes eps and min_samples (within eps), returns indicies of core samples and labels
        #print(core_samples, labels) #core_samples: indices of core samples, array[n_core_samples], labels: cluster labs for each pt, array[n_samples]
        dbscan_clus_labels = labels

        # Spectral Clustering
        spectral = SpectralClustering(n_clusters = k)
        #print(spectral.labels_)
        spectral_clus_labels = spectral.labels_

        kmeans_pred_labs = clusters_to_labels_voting(full_data, kmeans_clus_labels)
        #agglom_pred_labs = clusters_to_labels_voting(full_data, agglom_clus_labels)
        dbscan_pred_labs = clusters_to_labels_voting(full_data, dbscan_clus_labels)
        spectral_pred_labs = clusters_to_labels_voting(full_data, spectral_clus_labels)

        # evaluate returns a tuple: a list of scores and a matrix. ignoring matrix
        # only kmeans gives us the cluster centers, for others we will have to calculate them
    kmeans_k_score.append(evaluate(true_labs, kmeans_pred_labs, data, kmeans.cluster_centers_, k)[0]) # [0] because only taking the list of scores because im not gonna mess w matrix
    #agglom_k_score.append(evaluate(true_labs, agglom_pred_labs, data, compute_centroids(full_data, agglom_clus_labels), k)[0])
    dbscan_k_score.append(evaluate(true_labs, dbscan_pred_labs, data, compute_centroids(full_data, agglom_clus_labels), k)[0])
    spectral_k_score.append(evaluate(true_labs, spectral_pred_labs, data, compute_centroids(full_data, agglom_clus_labels), k)[0])


    print('K-Means Clustering Scores per K: ')
    print(kmeans_k_score) # print the table of scores

    print('Agglomerative Clustering Scores per K: ')
    #print(agglom_k_score)

    print('DBSCAN Clustering Scores per K: ')
    print(dbscan_k_score)
 
    print('Spectral Clustering Scores per K: ')
    print(spectral_k_score)
 
    method_names = ['SSE', 'Adj Rand', 'Norm Mut Info', 'Adj Mut Info', 'Homog', 'Completeness', 'V-Measure']


    #pd.DataFrame(agglom_k_score, columns = method_names), 

    return pd.DataFrame(kmeans_k_score, columns = method_names), pd.DataFrame(dbscan_k_score, columns = method_names), pd.DataFrame(spectral_k_score, columns = method_names) # just return ENTIRE MATRICES ( 4 PD DATA FRAMES OF N_K x N_METRICS)

if __name__ == '__main__':
    # Delcare target
    target = 'sex'

    # Load the data here
    mms = preprocessing.MinMaxScaler()
    full_data = pd.read_csv('data/clean_census_income.csv')

    # perform
    km, ag, db, spec = method_evaluation(full_data)
    print(km, ag, db, spec)

    # Stratify sampling to reduce data size. Cannot compute a distance matrix on all ~200000 instances
    # 'wt' = With Target
    # 'wot' = Without Target
    data_train, data_test = train_test_split(full_data, test_size=0.95, stratify=full_data.loc[:, target])
    full_data_names = full_data.columns
    scaled_data_train_wt = mms.fit_transform(data_train)
    scaled_data_test_wt = mms.fit_transform(data_test)
    scaled_data_train_wt = pd.DataFrame(scaled_data_train_wt, columns=full_data_names)

    # Separate out the data for various experiments here
    target_data = data_train.loc[:, target]
    data_train_wot = data_train.drop(columns=target)
    data_names = data_train_wot.columns
    scaled_data_train_wot = mms.fit_transform(data_train_wot)
    scaled_data_train_wot = pd.DataFrame(scaled_data_train_wot, columns=data_names)

    # Use Method_Evaluation to find optimal K (currently according to SILHUOETTE, but could be any metric)
    #kmeans_scores, agglom_scores, dbscan_scores, spectral_scores = method_evaluation(data, 'sex') # could make argument for which Metric u want optimal K for

    # Optimized K-means
    # can find best score here by looking at matrix
    #    plt.plot(x = kmeans_scores['k'], y = kmeans_scores['silhuoette'])
    #    plt.show()

    # Run the experiments
    k = 10

    # With target experiments
    centroids = []
    kmeans_wt = KMeans(n_clusters = k).fit(scaled_data_train_wt)
    pred_labs_wt = clusters_to_labels_voting(scaled_data_train_wt, kmeans_wt.labels_, target_data, target)
    internal_scores_list, cluster_corr = evaluate_internal(target_data, kmeans_wt.labels_, pred_labs_wt, scaled_data_train_wt, centroids)

    # Without target experiments
    kmeans_wot = KMeans(n_clusters = k).fit(scaled_data_train_wot)
    pred_labs_wot = clusters_to_labels_voting(scaled_data_train_wt, kmeans_wot.labels_, target_data, target)
    external_scores_list, cont_mat = evaluate_external(target_data, pred_labs_wot)
    print(internal_scores_list)
    print(cluster_corr)
    # NOTE: DO NOT RUN VIS, I think the heatmap is too big and crashes computers
    #visualization(scaled_data_train_wt, cluster_corr, target)
    print(external_scores_list)
    print(cont_mat)
    #visualization(scaled_data_train_wot, cont_mat, target)
    sys.exit()







    # LOL ALL THIS BELOW DOES NOT WORK

    visualization(data, kmeans.labels_)
    
    # Optimized Agglomerative Clustering
    agglom = AgglomerativeClustering(n_clusters = agglom_k)
    visualization(data, agglom.labels_)
    
    # Optimized DBSCAN Clustering
    _, dbscan_labels = DBSCAN(data, eps = .05, min_samples = dbscan_n) #this .05 needs to match the one used in method_evulation(), which we should experiment with
    visualization(data, dbscan_labels)

    # ADV TOPIC: Optimized Spectral Clustering
    spectral = SpectralClustering(n_clusters = spectral_k)
    visualization(data, spectral.labels_)

    print('K-Means Clustering Scores per K: ')
    print(kmeans_k_score) # print the table of scores
    pyplot.plot(x = range(1, 101), y = kmeans_k_score[7]) # print some cute little graph of score per K?
    pyplot.show()

    # CAN MAKE SKREE PLOTS HERE AS PART OF CLUSTERING ANALYSIS
    # COULD CALL THE VISUALIZATIONS FUNCTION IN HERE TOO

    print('Agglomerative Clustering Scores per K: ')
    print(agglom_k_score)
    pyplot.plot(x = range(1, 101), y = agglom_k_score[7])
    pyplot.show()

    # CAN MAKE SKREE PLOTS HERE AS PART OF CLUSTERING ANALYSIS
    # COULD CALL THE VISUALIZATIONS FUNCTION IN HERE TOO

    print('DBSCAN Clustering Scores per K: ')
    print(dbscan_k_score)
    pyplot.plot(x = range(1, 101), y = dbscan_k_score[7])
    pyplot.show()

    # CAN MAKE SKREE PLOTS HERE AS PART OF CLUSTERING ANALYSIS
    # COULD CALL THE VISUALIZATIONS FUNCTION IN HERE TOO

    print('Spectral Clustering Scores per K: ')
    print(spectral_k_score)
    pyplot.plot(x = range(1, 101), y = spectral_k_score[7])
    pyplot.show()

    # CAN MAKE SKREE PLOTS HERE AS PART OF CLUSTERING ANALYSIS
    # COULD CALL THE VISUALIZATIONS FUNCTION IN HERE TOO

    print('==============================================')

    #NOTE: This needs to be fixed. Busted and commented out for now
#    kmeans_best_k = np.argmin(kmeans_k_score[].silhuoette_scores) # "Which K got the lowest silhuoette score for this method?" <- silhguoete could be argument
#    agglom_best_k = np.argmin(agglom_k_score[].silhuoette_scores)
#    dbscan_best_k = np.argmin(dbscan_k_score[].silhuoette_scores)
#    spectral_best_k = np.argmin(spectral_k_score[].silhuoette_scores)
#
#
#    print('Optimal K selected per method selected by Silhuoette: ')
#    print()
#    print('Kmeans Clustering selects K = ', kmeans_best_k)
#
#    print('Agglom Clustering selects K = ', agglom_best_k)
#
#    print('DBSCAN Clustering selects K = ', dbscan_best_k)
#
#    print('Spectral Clusters selects K = ', spectral_best_k)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    for col in range(0, cm.shape[0]):
        if sum(row[col] for row in cm) == 0:
            temp = cm[col, col]
            cm[:, col] = -1
            cm[col, col] = temp
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set_aspect(1)
#    ax.set(xticks=np.arange(cm.shape[1]),
#           yticks=np.arange(cm.shape[0]),
    ax.set(xticks=range(cm.shape[1]),
           yticks=range(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

