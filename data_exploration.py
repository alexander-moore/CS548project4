#! /usr/bin/env python3
# data_exploration

import pandas as pd
import numpy as np
import seaborn as sns
import sys

np.set_printoptions(threshold=sys.maxsize)

data = pd.read_csv('clean_census_income.csv')
#data = data.iloc[:,0:20]
#data = pd.read_csv('log_cci.csv')
#print(data)

print(data.dtypes)

# all of the objects need to be one-hotted (in data setup)

# redundant columns?

ls = list(data['income'])

#print(ls)

counter = 0

for item in ls:
	if item == ' - 50000':
		counter += 1

print(counter / len(ls))

#print(data.describe())

desc = data.describe()
it = desc.shape[1]

for i in range(it):
	print(desc[desc.columns[i]])

cor_mat = data.corr()

names = data.columns.values

print(names)

#cor_mat.style.background_gradient()

print(cor_mat)

#sns.pairplot(data)


grumbo = data.isnull().sum(axis = 0)

for ele in grumbo:
	print(ele)