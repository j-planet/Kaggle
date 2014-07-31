import sys
sys.path.extend('/home/jj/code/Kaggle/Fire')

import pandas
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression, Ridge

from Kaggle.utilities import plot_histogram, plot_feature_importances
from globalVars import *


def process_data(dataFpath, impute, fieldsToUse=None):
    """
    reads data, processes discrete columns, imputes
    :param dataFpath: path to the data file
    :param impute: whether to impute the data
    :param fieldsToUse: if None use all the fields
    :return: x_data, y_data (None if not training data), ids (all np.arrays), columns
    """
    data = pandas.read_csv(dataFpath)

    ids = data['id']
    x_data = data

    # check if it's training data
    if 'target' in data.columns:
        y_data = data['target']
        del x_data['target']
    else:
        y_data = None

    # delete unused columns
    for col in ['id', 'dummy']:
        del x_data[col]

    # record columns
    columns = x_data.columns

    # handle ordinal continuous columns
    x_data[ORDINAL_CONT_COLS + DISCRETE_COLS] = x_data[ORDINAL_CONT_COLS + DISCRETE_COLS].replace(to_replace='Z', value=np.nan)

    # code discrete columns
    for col in DISCRETE_COLS:
        for oldVal, newVal in DISCRETE_COLS_LOOKUP[col].iteritems():
            x_data[col] = x_data[col].replace(to_replace=oldVal, value=newVal)

    if fieldsToUse is not None:
        x_data = x_data[fieldsToUse]
        columns = fieldsToUse

    if impute:
        print 'imputing x data'
        imp = Imputer()
        x_data = imp.fit_transform(np.array(x_data))

    return np.array(x_data), np.array(y_data), np.array(ids), columns


if __name__ == '__main__':

    # print 'about to plot feature importances'
    # plot_feature_importances(x_train, np.array(y_train), x_train.columns, numTopFeatures=0.85, numEstimators=50)


    # ================== train ==================
    print '================== train =================='
    x_train, y_train, _, columns_train = process_data('/home/jj/code/Kaggle/Fire/Data/train.csv', impute=True, fieldsToUse=FIELDS_20)

    # clf = GradientBoostingRegressor(loss='quantile', learning_rate=0.02, n_estimators=100, subsample=0.9)
    # clf = LogisticRegression()
    clf = Ridge(alpha=0.1)

    clf.fit(np.array(x_train), np.array(y_train))

    # ================== predict ==================
    print '================== predict =================='
    x_test, _, ids_pred, _ = process_data('/home/jj/code/Kaggle/Fire/Data/test.csv', impute=True, fieldsToUse=columns_train)
    pred = clf.predict(x_test)
    pandas.DataFrame({'id': ids_pred, 'target': pred}).\
        to_csv('/home/jj/code/Kaggle/Fire/submissions/20fieldsRidge.csv', index=False)

