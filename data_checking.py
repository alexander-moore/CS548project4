#! /usr/bin/env python3
# visualizations

#load data


# show MSE as function of k-fold (for tree regression lets say?)


#! /usr/bin/env python3

# linear regression
# age as regression target

#For regression tasks: use correlation coefficient AND any subset of the following error metrics
# that you find appropriate: mean-squared error, root mean-squared error, mean absolute error, 
# relative squared error, root relative squared error, and relative absolute error

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import tree
import sklearn.metrics
import math
from statistics import mean
from sklearn.model_selection import train_test_split
import re
from sklearn.model_selection import KFold
import time
import matplotlib.pyplot as plt

data = pd.read_csv('clean_census_income.csv')

ages = data['age']
incs = data['income']

plt.hist(ages)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.show()
plt.hist(incs)
plt.xlabel('Income')
plt.ylabel('Count')
plt.title('Income Distribution')
plt.show()

grep = data.iloc[:,2:8]

print(grep.describe())
