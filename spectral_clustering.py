#! /usr/bin/env python3
# spectral clustering

from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA

import sklearn.metrics as skm
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
    silhuoette = skm.silhuoette_score(data, cluster_labels)
    
    # Correlation Cluster Validity
    prox_mat = PROX???(data)
    match_mat = MATCH???(data)
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
def method_evaluation(data, method = SpectralClustering(), target = 'sex'):

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

if __name__ == '__main__':

    main(k)

    # lets just load and scale the data
    data = pd.read_csv('data/clean_census_income.csv')
    mms = sklearn.preprocessing.MinMaxScaler()
    data = mms.fit_transform(data)

    # Kmeans Clustering
    learn_k = method_evaluation(data, Kmeans)

    classes = do_kmeans(data, learn_k)

    visualization(data, classes)
    #
    # Hierarchical Clustering (Agglomerative Clustering)
    #
    # DBSCAN Clustering

