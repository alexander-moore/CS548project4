# data preprocessing
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import pandas as pd

data = pd.read_csv('data/clean_census_income.csv')
print(data.shape)

#third_data = data.loc[0:data.shape[0]/5, :]
#print(third_data.shape)

pca_embed = PCA(n_components = 2)

pcad = pd.DataFrame(pca_embed.fit_transform(data))
print(pcad.shape)

pcad.to_csv('data/pcad_data.csv')