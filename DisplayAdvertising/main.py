__author__ = 'jennyjin'

import os, sys
sys.path.append('/Users/jennyjin/K/DisplayAdvertising')

import pandas
import numpy as np
from pprint import pprint
from matplotlib import pyplot as plt

from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression

from utilities import print_missing_values_info, plot_histogram, plot_feature_importances
from GlobalVars import *


# --------- read data -------------
trainData = pandas.read_csv(os.path.join(DATADIR, 'smallTrain.csv'))
train_Y = trainData['Label']
train_X = trainData[['I' + str(i) for i in range(1, 14)]]

temp = pandas.isnull(trainData).sum()


print_missing_values_info(trainData)
imp = Imputer()
train_X = imp.fit_transform(train_X)
plot_feature_importances(train_X, train_Y, ['I' + str(i) for i in range(1, 14)], 0.99)

# --------- train -------------
clf = LogisticRegression()
clf.fit(train_X, train_Y)


# --------- predict -------------
testData = pandas.read_csv(os.path.join(DATADIR, 'test.csv'))
test_X = testData[['I' + str(i) for i in range(1, 14)]]
test_X = imp.transform(test_X)
clf.predict(test_X)
pred = clf.predict_proba(test_X)[:, 1]  # probability of being 1

pandas.DataFrame({'Id': testData['Id'], 'Predicted': pred}).to_csv(os.path.join(SUBMISSIONDIR, 'smallTrain.csv'), index=False)