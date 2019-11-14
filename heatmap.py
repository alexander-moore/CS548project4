# heatmap
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('tiny_cci.csv')
prox = pd.read_csv('tiny_prox_mat.csv')

# heatmap of similarity matrix, sorted by cluster
kmeans = KMeans(n_clusters = 8).fit(data)

prox['labels'] = kmeans.labels_

sorted_prox = prox.sort_values(by = ['labels'])

del sorted_prox['labels']

sns.heatmap(sorted_prox)
plt.show()