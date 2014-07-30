# import sys
# sys.path.extend('/home/jj/code/Kaggle/Fire')

import pandas
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import Imputer

from Kaggle.utilities import plot_histogram, plot_feature_importances
from globalVars import *

trainData = pandas.read_csv('/home/jj/code/Kaggle/Fire/Data/train.csv')
y_train = trainData['target']
x_train = trainData

# delete unused columns
for col in ['target', 'id', 'dummy']:
    del x_train[col]

# handle ordinal continuous columns
x_train[ORDINAL_CONT_COLS + DISCRETE_COLS] = x_train[ORDINAL_CONT_COLS + DISCRETE_COLS].replace(to_replace='Z', value=np.nan)

# code discrete columns
for col in DISCRETE_COLS:
    for oldVal, newVal in DISCRETE_COLS_LOOKUP[col].iteritems():
        x_train[col] = x_train[col].replace(to_replace=oldVal, value=newVal)

# plot_histogram(np.array(y_train[y_train > 0]), 25)
# print 'about to plot feature importances'
# plot_feature_importances(x_train, np.array(y_train), x_train.columns, numTopFeatures=0.85, numEstimators=50)

# --------------- train ---------------------
print '--------------- train ---------------------'
imp = Imputer()
x_train = imp.fit_transform(np.array(x_train))

clf = GradientBoostingRegressor(loss='quantile', learning_rate=0.05, n_estimators=100, subsample=1)
clf.fit(np.array(x_train), np.array(y_train))

# --------------- predict ---------------------
print '--------------- predict ---------------------'
testData = pandas.read_csv('/home/jj/code/Kaggle/Fire/Data/test.csv')
ids_pred = testData['id']

for col in ['id'] + DISCRETE_COLS:
    if col in testData.columns:
        del testData[col]
pred = clf.predict(imp.transform(np.array(testData)))
pandas.DataFrame({'id': ids_pred, 'target': pred}).\
    to_csv('/home/jj/code/Kaggle/Fire/Submissions/fullTrainInitSubmission.csv', index=False)