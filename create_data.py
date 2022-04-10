import random

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import csv
from sklearn.model_selection import train_test_split

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

path_train_input = r'data\train.csv'
path_test_input = r'data\test.csv'

pd.set_option('display.max_columns', None)
train_df = pd.read_csv(path_train_input)
test_df = pd.read_csv(path_test_input)
combine = [train_df, test_df]
columns = train_df.columns.values
# print(train_df.info())
# print(train_df.describe())
# print(train_df.describe(include=['O']))
# print(train_df[['LotArea', 'SalePrice']].groupby(['LotArea'], as_index=False).mean().sort_values(by='SalePrice', ascending=False))

name = 'MoSold'
train_df.plot.scatter(x=name, y='SalePrice')

print(train_df[name])

for dataset in combine:
    dataset.loc[dataset['LotArea'] > 25000, 'LotArea'] = 25000
#train_df.plot.scatter(x='LotArea', y='SalePrice')
plt.show()
