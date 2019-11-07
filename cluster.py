#! /usr/bin/env python3
# cluster.py
import sklearn.metrics as skm

from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fowlkes_mallows_score
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

def evaluate(true_labs, pred_labs, data, centroids, k): # evaluate WRT sex

    SSE = need_SSE_func(centers, pred_labs) # centers are given by the methods
	ajd_rand = skm.adjusted_rand_score(true_labs, pred_labs)
	norm_info = skm.normalized_mutual_info_score(true_labs, pred_labs)
	adj_info = skm.adjusted_mutual_info_score(true_labs, pred_labs)
	homog = skm.homogeneity_score(true_labs, pred_labs)
	complete = skm.completeness_score(true_labs, pred_labs)
	v_measure = skm.v_measure_score(true_labs, pred_labs)
	cont_mat = skm.cluster.contingency_matrix(true_labs, pred_labs)

    silhuoette = skm.silhuoette_score(data, cluster_labs) # need to verify arguments on this
    
    # Correlation Cluster Validity
    # pairwise row-distances matrix (entry i,j is the distance from point i to point j of data)
    prox_mat = pd.DataFrame(distance_matrix(data.values, data.values), index = data.index, columns = data.index)
    # match matrix: entry i,j is 1 if observation i and j share a cluster, 1 otherwise
    match_mat = MATCH???(data)
    # correlation of elements of prox_mat and match_mat, ideally close to -1:
    clus_cor = np.corr(prox_mat, match_mat)

	return [k, SSE, adj_rand, norm_info, adj_info, homog, complete, v_measure, silhuoette, clus_cor], cont_mat


def visualization(data, classes):

    ## Full-Dimension Visualizations
    # similarity heatmap
    data = sort_data(by = classes) ??

    sim_mat = similarity_matrix(data)
    jeat = sns.heatmap(sim_mat)
    heat.show()

    ## Reduced Dimension Visualizations
    # Convert Data to PC2
    pca_df = pd.DataFrame(pca_data, columns = ['PC_1', 'PC_2'])

    ## Visualize
    plt.plot(x = PC1, y = PC2, color = classes, legend = classes)
    plt.show()


# this will turn arbitrary cluster labelling [clus1, clus2, ..] into the same type as our target (1 male 0 female) by getting the mode target of each cluster
def clusters_to_labels_voting(data = full_data, labels, target = 'sex'):

    method_pred_labs = []

    for clus in set(labels): # for each cluster

        subset = data[target] == clus # just get the data in that cluster
        true_label_mode = subset.mode[target] # get the mode target in that cluster
        subset_pred_vec = [true_label_mode] * subset.shape[0] # each element of subset gets the same modal label
        method_pred_labs.extend(subset_pred_vec) # use extend to concatenate this ckuster's output

    return method_pred_labs


# method_evaluation should create, display, and select best scoring K across methods
def method_evaluation(data, target = 'sex', optimization_metric = 'Silhuoette'):

    full_data = data.copy() # we need full data (no drop target) for the mode-voting

    target_data = data.loc[:, target]
    target_names = [str(x) for x in target_data.unique().tolist()] # i still dont know what this is

    true_labs = target_data

    data = data.drop(columns=target)

    for k in range(1, 101):

        # Kmeans
        kmeans = KMeans(n_clusters = k).fit(data) #random state = 0 ?
        print(kmeans.labels_)
        kmeans_clus_labels = kmeans.labels_
        print(kmeans.predict(data))
        print(kmeans.cluster_centers_)

        # Agglom
        agglom = AgglomerativeClustering(n_clusters = k).fit(data)
        print(agglom.labels_)
        agglom_clus_labels = agglom.labels_

        # DBSCAN
        # eps is the ball radius around points. remember our space is in high dimension, but scaled to 0,1. so this is fucked
        core_samples, labels = DBSCAN(data, eps = .05, min_samples = k) #takes eps and min_samples (within eps), returns indicies of core samples and labels
        print(core_samples, labels) #core_samples: indices of core samples, array[n_core_samples], labels: cluster labs for each pt, array[n_samples]
        dbscan_clus_labels = labels

        # Spectral Clustering
        spectral = SpectralClustering(n_clusters = k)
        print(spectral.labels_)
        spectral_clus_labels = spectral.labels_

        kmeans_pred_labs = clusters_to_labels_voting(full_data, kmeans_clus_labels)
        agglom_pred_labs = clusters_to_labels_voting(full_data, agglom_clus_labels)
        dbscan_pred_labs = clusters_to_labels_voting(full_data, dbscan_clus_labels)
        spectral_pred_labs = clusters_to_labels_voting(full_data, spectral_clus_labels)

        # evaluate returns a tuple: a list of scores and a matrix. ignoring matrix
        # only kmeans gives us the cluster centers, for others we will have to calculate them
        kmeans_k_score.append(evaluate(true_labs, kmeans_pred_labs, data, kmeans.cluster_centers_, k)[0]) # [0] because only taking the list of scores because im not gonna mess w matrix
        agglom_k_score.append(evaluate(true_labs, agglom_pred_labs, data, compute_centroids(full_data, agglom_clus_labels), k)[0])
        dbscan_k_score.append(evaluate(true_labs, dbscan_pred_labs, data, compute_centroids(full_data, agglom_clus_labels), k)[0])
        spectral_k_score.append(evaluate(true_labs, spectral_pred_labs, data, compute_centroids(full_data, agglom_clus_labels), k)[0])

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

    kmeans_best_k = np.argmin(kmeans_k_score[].silhuoette_scores) # "Which K got the lowest silhuoette score for this method?" <- silhguoete could be argument
    agglom_best_k = np.argmin(agglom_k_score[].silhuoette_scores)
    dbscan_best_k = np.argmin(dbscan_k_score[].silhuoette_scores)
    spectral_best_k = np.argmin(spectral_k_score[].silhuoette_scores)


    print('Optimal K selected per method selected by Silhuoette: ')
    print()
    print('Kmeans Clustering selects K = ', kmeans_best_k)

    print('Agglom Clustering selects K = ', agglom_best_k)

    print('DBSCAN Clustering selects K = ', dbscan_best_k)

    print('Spectral Clusters selects K = ', spectral_best_k)

    return kmeans_best_k, agglom_best_k, dbscan_best_k, spectral_best_k

if __name__ == '__main__':

    main(k)

    # lets just load and scale the data here
    data = pd.read_csv('data/clean_census_income.csv')
    mms = sklearn.preprocessing.MinMaxScaler()
    data = mms.fit_transform(data)

    # Use Method_Evaluation to find optimal K (currently according to SILHUOETTE, but could be any metric)
    kmeans_k, agglom_k, dbscan_n, spectral_k = method_evaluation(data, 'sex') # could make argument for which Metric u want optimal K for

    # Optimized K-means
    kmeans = KMeans(n_clusters = kmeans_k).fit(data)
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





#def main(k):
#    data, target = load_digits(return_X_y=True)
#    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#
#    # K-means clustering
#    kmeans = KMeans(n_clusters=k).fit(data)
#    pred_target = np.zeros(target.shape)
#    for i in range(0, k):
#        cluster = []
#        index_list = []
#        for j in range(0, data.shape[0]):
#            if kmeans.predict([data[j]]) == i:
#                cluster.append(target[j])
#                index_list.append(j)
#        #print('Cluster {} Results:'.format(max(set(cluster), key = cluster.count)))
#        print('Cluster Results:')
#        predicted_reg = max(set(cluster), key = cluster.count)
#        print('  Predicted Representative: {}'.format(predicted_reg))
#        print('  Actual Classes:')
#        for num in range(0, k):
#            print('    {}\'s: {}'.format(num, cluster.count(num)))
#        print('  Cluster Rep. Indexes: {}'.format(index_list))
#        for index in index_list:
#            pred_target[index] = predicted_reg
#    print('  Fowlkes-Mallows score: {}'.format(fowlkes_mallows_score(target, pred_target)))
#    plot_confusion_matrix(target, pred_target, class_names, title='K-means Confusion Matrix')
#    plt.show()

    # Agglomerative clustering
#    cluster_results = AgglomerativeClustering(n_clusters=k).fit_predict(data)
#    pred_target = np.zeros(target.shape)
#    for i in range(0, k):
#        cluster = []
#        index_list = []
#        for j in range(0, len(cluster_results)):
#            if cluster_results[j] == i:
#                cluster.append(target[j])
#                index_list.append(j)
#        #print('Cluster {} Results:'.format(max(set(cluster), key = cluster.count)))
#        print('Cluster Results:')
#        predicted_reg = max(set(cluster), key = cluster.count)
#        print('  Predicted Representative: {}'.format(predicted_reg))
#        print('  Actual Classes:')
#        for num in range(0, k):
#            print('    {}\'s: {}'.format(num, cluster.count(num)))
#        print('  Cluster Rep. Indexes: {}'.format(index_list))
#        for index in index_list:
#            pred_target[index] = predicted_reg
#    print('  Fowlkes-Mallows score: {}'.format(fowlkes_mallows_score(target, pred_target)))
#    plot_confusion_matrix(target, pred_target, class_names, title='Agglomerative Confusion Matrix')
#    plt.show()




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
