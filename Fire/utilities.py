import os
import pandas
import numpy as np
from scipy import stats
from pprint import pprint
from copy import copy

from sklearn.cross_validation import KFold

from Kaggle.utilities import jjcross_val_score

from evaluation import normalized_weighted_gini
from globalVars import *


def process_data(dataFpath, impute, fieldsToUse=None, imputeDataDir=None, imputeStrategy=None):
    """
    reads data, processes discrete columns, imputes
    :param dataFpath: path to the data file
    :param impute: whether to impute the data
    :param fieldsToUse: if None use all the fields
    :param imputeStrategy: can only be one of {'mean', 'median', 'mode'}
    :return: x_data, y_data (None if not training data), ids (all np.arrays), columns, weights, y_class
    """

    data = pandas.read_csv(dataFpath)

    ids = data['id']
    weights = data['var11']
    x_data = data

    # check if it's training data
    if 'target' in data.columns:
        y_data = data['target']
        del x_data['target']
    else:
        y_data = None
    y_data = np.array(y_data, dtype=np.float)

    # make classification y
    y_class = copy(y_data)
    y_class[y_class > 0] = 1
    y_class = np.array(y_class, dtype=np.int)

    # delete unused columns
    for col in NON_PREDICTOR_COLS:
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
        assert imputeDataDir is not None and imputeStrategy in {'mean', 'median', 'mode'}, 'Invalid imputing setup.'

        imputeData = pandas.read_csv(os.path.join(imputeDataDir, 'impute_' + imputeStrategy + '.csv'))

        for col in x_data.columns:
            # x_data[col] = x_data[col].replace(to_replace=np.nan, value=imputeData[col])
            x_data[col][np.isnan(np.array(x_data[col], dtype=np.float))] = imputeData[col][0]

    return np.array(x_data, dtype=np.float), y_data, np.array(ids), columns, np.array(weights), y_class


def mode(l):
    """
    computes the mode of a list of numbers. if there are multiple, take their average
    :param l: list of numbers
    :return: a single number
    """

    return stats.mstats.mode(l)[0][0]


def make_column_2D(l):
    """
    make a column/row vector 2D. i.e. shape (n,) to shape (n,1)
    :param l: a np.array of shape (n,)
    :return:
    """

    return l.reshape(len(l), 1)


def convert_to_cdfs(y):
    """
    convert each value in y to its corresponding empirical cdf
    :param y: numpy array or list
    :return: a numpy array
    """

    y = np.array(y)
    return np.array([(y<=v).sum() for v in y])*1./len(y)


def gridSearch(clf, cvOutputFname, x_train, y_train, weights, num_folds = 10):
    print '================== Grid Search for the Best Parameter  =================='

    cvOutputFile = open(cvOutputFname, 'w')
    res = {}
    cvObj = KFold(len(y_train), n_folds=num_folds, shuffle=True, random_state=0)

    for tolerance in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]:
        for alpha in np.arange(0.01, 5, 0.025):
            print '>>>> alpha=', alpha, ', tolerance =', tolerance
            clf.set_params(alpha=alpha, tol=tolerance)
            scores = jjcross_val_score(clf, x_train, y_train, normalized_weighted_gini, cvObj, weights=weights,
                                       verbose=False)
            meanScore = np.mean(scores)
            stdScore = np.std(scores)
            s = 'alpha = %f, tolerance = %f, mean = %f, std = %f\n' % (alpha, tolerance, meanScore, stdScore)
            print s
            res[(alpha, tolerance)] = (meanScore, stdScore)
            cvOutputFile.write(s)
    print '>>>>>> Result sorted by mean score:'
    pprint(sorted(res.items(), key=lambda x: -x[1][0]))
    cvOutputFile.close()

    return res
