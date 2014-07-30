import pandas
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import Imputer

from Kaggle.utilities import plot_histogram

discreteCols = ['var%d' % i for i in range(1, 10)] + ['dummy']
trainData = pandas.read_csv('/home/jj/code/Kaggle/Fire/Data/train.csv')
y_train = trainData['target']
x_train = trainData
for col in ['target', 'id'] + discreteCols:
    del x_train[col]

# plot_histogram(np.array(y_train[y_train > 0]), 25)

# --------------- train ---------------------
imp = Imputer()
x_train = imp.fit_transform(np.array(x_train))

clf = GradientBoostingRegressor(loss='quantile', learning_rate=0.05, n_estimators=100, subsample=1)
clf.fit(np.array(x_train), np.array(y_train))

# --------------- predict ---------------------
testData = pandas.read_csv('/home/jj/code/Kaggle/Fire/Data/test.csv')
ids_pred = testData['id']

for col in ['id'] + discreteCols:
    if col in testData.columns:
        del testData[col]
pred = clf.predict(imp.transform(np.array(testData)))
pandas.DataFrame({'id': ids_pred, 'target': pred}).\
    to_csv('/home/jj/code/Kaggle/Fire/Submissions/smallTrainInitSubmission.csv', index=False)