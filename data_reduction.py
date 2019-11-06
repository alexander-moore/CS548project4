# data preprocessing
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


data = pd.read_csv('data/clean_census_income.csv')
print(data.shape)
#print(data.columns.values)

#no_sex_d = data.copy()
#del no_sex_d['sex']
#print(no_sex_d.shape)

#third_data = data.loc[0:data.shape[0]/5, :]
#print(third_data.shape)

#pca_embed = PCA(n_components = 2)
#pca_embed2 = PCA(n_components = 2)

#pcad = pd.DataFrame(pca_embed.fit_transform(data))
#print(pcad.shape)

#pcad.to_csv('data/pcad_data.csv')

#pcad_nosex = pd.DataFrame(pca_embed2.fit_transform(no_sex_d))
#print(pcad_nosex.shape)

#pcad_nosex.to_csv('data/pcad_nosex.csv')

mms = preprocessing.MinMaxScaler()
#data[data.columns] = mms.fit_transform(data[data.columns])
data = mms.fit_transform(data)

pd.DataFrame(data).to_csv('data/mms_cci.csv')

pca_full = PCA(n_components = 509)
pcad_full = pd.DataFrame(pca_full.fit_transform(data))

print(pca_full)
#print(pca_full.explained_variance_)
print(pca_full.explained_variance_ratio_[0:10])
print('we will use rotations later: ', pca_full.components_)

# plotting shows 12 to be a good number of components
#plt.plot(pca_full.explained_variance_ratio_)
#plt.show()
df = pd.read_csv('data/pca2_data.csv')

plt.plot(df['PC_1'], df['PC_2'])
plt.show()