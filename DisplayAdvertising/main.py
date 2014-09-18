__author__ = 'jennyjin'

import os, sys
sys.path.append('/Users/jennyjin/K/DisplayAdvertising')

import pandas
import numpy as np
from pprint import pprint
from matplotlib import pyplot as plt

from sklearn.preprocessing import Imputer, LabelEncoder, Normalizer
from sklearn.linear_model import LogisticRegression

from utilities import print_missing_values_info, plot_histogram, plot_feature_importances
from GlobalVars import *


# --------- read data -------------
ordCols = ['I' + str(i) for i in range(1, 14)]
catCols = ['C' + str(i) for i in range(1, 27)]
trainData = pandas.read_csv(os.path.join(DATADIR, 'smallTrain.csv'))

td = trainData
# td = pandas.read_csv(os.path.join(DATADIR, 'minTrain.csv'))
train_Y = td['Label']

ord_train = td[ordCols]
cat_train = td[catCols]
cat_train_num = np.empty(cat_train.shape, dtype=int)

for i in range(cat_train.shape[1]):
    le = LabelEncoder()
    cat_train_num[:,i] = le.fit_transform(cat_train.icol(i))

cat_train_num = pandas.DataFrame(data = cat_train_num, columns=cat_train.columns)

train_X = ord_train.join(cat_train_num)


# print_missing_values_info(trainData)
imp = Imputer()
train_X = imp.fit_transform(train_X)
normer = Normalizer()
train_X = normer.fit_transform(train_X)
plot_feature_importances(train_X, train_Y, np.array(ordCols + catCols), 0.99, num_jobs=7)

# --------- train -------------
# clf = LogisticRegression()
# clf.fit(train_X, train_Y)
#
#
# # --------- predict -------------
# testData = pandas.read_csv(os.path.join(DATADIR, 'test.csv'))
# test_X = testData[['I' + str(i) for i in range(1, 14)]]
# test_X = imp.transform(test_X)
# clf.predict(test_X)
# pred = clf.predict_proba(test_X)[:, 1]  # probability of being 1
#
# pandas.DataFrame({'Id': testData['Id'], 'Predicted': pred}).to_csv(os.path.join(SUBMISSIONDIR, 'smallTrain.csv'), index=False)