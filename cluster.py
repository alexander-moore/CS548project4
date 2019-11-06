#! /usr/bin/env python3
# cluster.py
import sklearn.metrics as skm

from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
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

def evaluate(true_labs, pred_labs, data = data, train_time): # evaluate WRT sex

	ajd_rand = skm.adjusted_rand_score(true_labs, pred_labs)
	norm_info = skm.normalized_mutual_info_score(true_labs, pred_labs)
	adj_info = skm.adjusted_mutual_info_score(true_labs, pred_labs)
	homog = skm.homogeneity_score(true_labs, pred_labs)
	complete = skm.completeness_score(true_labs, pred_labs)
	v_measure = skm.v_measure_score(true_labs, pred_labs)
	cont_mat = skm.cluster.contingency_matrix(true_labs, pred_labs)
    silhuoette = skm.silhuoette_score(data, cluster_labs)
    
    # Correlation Cluster Validity
    # pairwise row-distances matrix (entry i,j is the distance from point i to point j of data)
    prox_mat = pd.DataFrame(distance_matrix(data.values, data.values), index = data.index, columns = data.index)
    # match matrix: entry i,j is 1 if observation i and j share a cluster, 1 otherwise
    match_mat = MATCH???(data)
    # correlation of elements of prox_mat and match_mat, ideally close to -1:
    clus_cor = np.corr(prox_mat, match_mat)

	return [adj_rand, norm_info, adj_info, homog, complete, v_measure, silhuoette, clus_cor, train_time], cont_mat

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

# method_evaluation should return metrics over a large potential number of K's
def method_evaluation(data, method, target = 'sex'):

    score_matrix = np.zeros((50, 9))

    target_data = data.loc[:, target]
    target_names = [str(x) for x in target_data.unique().tolist()]
    data = data.drop(columns=target)

    index = 1

    t1 = time.perf_counter()

    for k in range(1, 51):
        obs_clus_labels = method.fit_predict(data)

    train_time = time.perf_counter() - t1
        
    #pred_target = method.predict(data_test)
    pred_label = get_cluster_modes()

    metrics_row, _ = evaluate(true_labs, pred_labs, data, train_time)
          
    for i in range(len(metrics_row)):
        score_matrix.iloc[index-1, i] = metrics_row[i]

    index += 1

    print(score_matrix)

    return score_matrix, recc__k

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

if __name__ == '__main__':

    main(k)

    # lets just load and scale the data here
    data = pd.read_csv('data/clean_census_income.csv')
    mms = sklearn.preprocessing.MinMaxScaler()
    data = mms.fit_transform(data)

    # Kmeans Clustering
    learn_k = method_evaluation(data, Kmeans)

    classes = do_kmeans(data, learn_k)

    visualization(data, classes)
    #
    # Hierarchical Clustering (Agglomerative Clustering)
    learn_k = method_evaluation(data, AgglomerativeClustering(variable_args))

    classes = do_kmeans(data, learn_k)

    visualization(data, classes)
    
    # DBSCAN Clustering
    learn_k = method_evaluation(data, DBSCAN(variable_args))

    classes = do_kmeans(data, learn_k)

    visualization(data, classes)

    # ADV TOPIC: Spectral Clustering
    learn_k = method_evaluation(data, SpectralClustering(variable_args))

    classes = do_kmeans(data, learn_k)

    visualization(data, classes)






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
