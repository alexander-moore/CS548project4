# heatmap
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv('tiny_cci.csv')
prox = pd.read_csv('tiny_prox_mat.csv')

# heatmap of similarity matrix, sorted by cluster
kmeans = KMeans(n_clusters = 8).fit(data)

sorted = prox.sort_values(by = kmeans.labels_)

map = sns.heatmap(tiny_prox_mat)
map.show()