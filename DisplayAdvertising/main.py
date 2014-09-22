__author__ = 'jennyjin'

import sys
sys.path.append('/Users/jennyjin/K/DisplayAdvertising')

import pandas
import numpy as np
from pprint import pprint
from matplotlib import pyplot as plt

from sklearn.preprocessing import Imputer, LabelEncoder, Normalizer
from sklearn.linear_model import LogisticRegression

from utilities import print_missing_values_info, plot_histogram, plot_feature_importances
from GlobalVars import *


def read_train_data(fname):
    """
    assume the file is training data
    :param fname: no path. just the name. e.g. 'train.csv'
    :return: X, Y, catEncoders, imp, normer, columns for X
    """

    data = pandas.read_csv(os.path.join(DATADIR, fname))
    Y = data['Label']

    ord_data = data[ORD_COLS]

    # --- process categorical data
    catEncoders = {}
    cat_data = data[CAT_COLS]
    cat_data_num = np.empty(cat_data.shape, dtype=int)

    for i in range(cat_data.shape[1]):
        le = LabelEncoder()
        cat_data_num[:, i] = le.fit_transform(np.array(cat_data.icol(i), dtype=np.str))
        catEncoders[CAT_COLS[i]] = le

    cat_train_num = pandas.DataFrame(data = cat_data_num, columns=cat_data.columns)

    X = ord_data.join(cat_train_num)
    imp = Imputer()
    X = imp.fit_transform(X)
    normer = Normalizer()
    X = normer.fit_transform(X)

    return X, Y, catEncoders, imp, normer, ORD_COLS + CAT_COLS


def read_test_data(fname, catEncoders, imp, normer):
    """
    assume the file is test data
    :param fname: no path. just the name. e.g. 'test.csv'
    :param catEncoders: dictionary of label encoders {col: label encoder for col}
    :param imp: imputer for X, got from training data
    :param normer: normalizer for X, got from training data
    :return: X, ids
    """

    data = pandas.read_csv(os.path.join(DATADIR, fname))
    ids = data['Id']

    ord_data = data[ORD_COLS]

    # --- process categorical data
    cat_data = data[CAT_COLS]
    cat_data_num = np.empty(cat_data.shape, dtype=int)

    for i in range(cat_data.shape[1]):
        le = catEncoders[CAT_COLS[i]]
        cat_data_num[:,i] = le.transform(np.array(cat_data.icol(i), dtype=np.str))

    cat_train_num = pandas.DataFrame(data=cat_data_num, columns=cat_data.columns)

    X = ord_data.join(cat_train_num)
    X = normer.transform(imp.transform(X))

    return X, ids

# --------- read data -------------
train_X, train_Y, catEncoders, imp, normer, xCols = read_train_data('train.csv')
# print_missing_values_info(pandas.DataFrame(train_X, columns=xCols))
# plot_feature_importances(train_X, train_Y, np.array(xCols), 0.99, num_jobs=7)

# --------- train -------------
print '--------- train -------------'
clf = LogisticRegression()
clf.fit(train_X, train_Y)


# --------- predict -------------
print '--------- predict -------------'
test_X, test_ids = read_test_data('test.csv', catEncoders, imp, normer)

clf.predict(test_X)
pred = clf.predict_proba(test_X)[:, 1]  # probability of being 1

pandas.DataFrame({'Id': test_ids, 'Predicted': pred}).to_csv(os.path.join(SUBMISSIONDIR, 'allFields.csv'), index=False)