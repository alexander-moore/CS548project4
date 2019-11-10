#little maker
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing

mms = preprocessing.MinMaxScaler()

data = pd.read_csv('data/clean_census_income.csv')
mms_data = mms.fit_transform(data)

tiny_data, _ = train_test_split(mms_data, test_size=0.95) # trying 10% for now
tiny_prox_mat = pd.DataFrame(pairwise_distances(tiny_data))

tiny_data.to_csv('tiny_cci.csv', index = False)
tiny_prox_mat.to_csv('tiny_prox_mat.csv', index = False)
