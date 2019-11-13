import pandas as pd
from sklearn.manifold import Isomap
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('tiny_cci.csv')
mms_data = MinMaxScaler().fit_transform(data)

def visualization(data, cluster_labels):
    
    ## ISOMAP Reduced
    isom = Isomap(n_components = 2)
    isom_data = isom.fit_transform(data)

    isom_df = pd.DataFrame(data = isom_data, columns = ['Dim1', 'Dim2'])

    plt.scatter(isom_df['Dim1'], isom_df['Dim2'], c = cluster_labels, alpha = .5)
    plt.xlabel('Dim1')
    plt.ylabel('Dim2')
    plt.title('DBSCAN of eps=.1, min_samples = 5 via ISOMAP')
    plt.show()

    ## Reduced Dimension Visualizations
    pca = MDS(n_components=2)
    pca_data = pca.fit_transform(data)

    pca_df = pd.DataFrame(data = pca_data, columns = ['PC1', 'PC2'])

    # Visualize
    plt.scatter(pca_df['PC1'], pca_df['PC2'], c = cluster_labels, alpha = .5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('THIS IS MDS THIS IS MDS')
    plt.show()

dbscan = DBSCAN(eps = 2.3).fit(mms_data)
visualization(mms_data, dbscan.labels_)