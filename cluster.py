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

def evaluate(true_labs, pred_labs, X, labels): # evaluate WRT income, sex

	ajd_rand = skm.adjusted_rand_score(true_labs, pred_labs)
	norm_info = skm.normalized_mutual_info_score(true_labs, pred_labs)
	adj_info = skm.adjusted_mutual_info_score(true_labs, pred_labs)
	homog = skm.homogeneity_score(true_labs, pred_labs)
	complete = skm.completeness_score(true_labs, pred_labs)
	v_measure = skm.v_measure_score(true_labs, pred_labs)
	silhuoette = skm.silhuoette_score(X, labels) # what ?
	cont_mat = skm.cluster.contingency_matrix(true_labs, pred_labs)

	return [adj_rand, norm_info, adj_info, homog, complete, v_measure, silhuoette], cont_mat

def visualization(data, classes):

    ## Convert Data to PC2
    mms = preprocessing.MinMaxScaler()
    data = mms.fit_transform(data)

    pca = PCA(n_components = 2).
    pca_data = pca.fit_transform(data)

    pca_df = pd.DataFrame(pca_data, columns = ['PC_1', 'PC_2'])

    ## Visualize

    plot(x = PC1, y = PC2, col = classes)


    pass

def evaluation(data, target, method, n_splits):
    # separate into test and train
    target_data = data.loc[:, target]
    target_names = [str(x) for x in target_data.unique().tolist()]
    data = data.drop(columns=target)

    # rescale the predictor columns
    min_max_scaler = preprocessing.MinMaxScaler()
    data[data.columns] = min_max_scaler.fit_transform(data[data.columns])
    
    #data = data.rename(columns=lambda x: re.sub('&', '&amp;', x))

    # determine prediction type
#    if isinstance(method, sklearn.neural_network.multilayer_perceptron.MLPClassifier):
#        classification = True
#        kf = StratifiedKFold(n_splits=k)
#        score_matrix = pd.DataFrame(np.zeros(shape = (k, 10)), columns=['Loss', 'Accuracy', 'Precision', 'Recall', 'ROC AUC', 'TN', 'FP', 'FN', 'TP', 'Time']) # needs to be the right number of metrics

    index = 1

    kf = StratifiedKFold(n_splits=n_splits)
    for train_index, test_index in kf.split(data, target_data):
        print('Split: {}'.format(index))
        #print('Train:', train_index, 'Test:', test_index)
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]
        target_train, target_test = target_data.iloc[train_index], target_data.iloc[test_index]

        t1 = time.perf_counter()
        method.fit(data_train, target_train)
        train_time = time.perf_counter() - t1
        
        pred_target = method.predict(data_test)
        #print(pred_target)


        # save iteration method performance:
        if classification:
            #print('score: {}'.format(score))
            results_prob = method.predict_proba(data_test)
#            print('Layers:')
#            print(method.n_layers_)
            predict_prob = results_prob[:, 1] # results_prob is something we can get form classifier. you have to ask for it: usually it just predicts classes
            #print(results_prob)
            #print('Predict_prob: {}'.format(predict_prob))
            metrics_row = classification_metrics(method.loss_, target_test.values.tolist(), pred_target.tolist(), predict_prob, train_time)
            #print(score_matrix.shape)
            #print(len(metrics_row))
            for i in range(len(metrics_row)):
                score_matrix.iloc[index-1, i] = metrics_row[i]

        index += 1

    print(score_matrix)
    score_list = df_to_scores(score_matrix)

    return score_matrix, score_list

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
    k = 10
    main(k)
    data = pd.read_csv('data/clean_census_income.csv')

    # Kmeans Clustering
    learn_k = evaluation(data)

    classes = do_kmeans(data, learn_k)

    visualization(data, classes)
    #
    # Hierarchical Clustering (Agglomerative Clustering)
    #
    # DBSCAN Clustering

# sklearn.manifold.MDS
