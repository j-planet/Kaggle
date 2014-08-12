import os
import pandas
import numpy as np
from scipy import stats

from globalVars import *


def process_data(dataFpath, impute, fieldsToUse=None, imputeDataDir=None, imputeStrategy=None):
    """
    reads data, processes discrete columns, imputes
    :param dataFpath: path to the data file
    :param impute: whether to impute the data
    :param fieldsToUse: if None use all the fields
    :param imputeStrategy: can only be one of {'mean', 'median', 'mode'}
    :return: x_data, y_data (None if not training data), ids (all np.arrays), columns, weights
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

    return np.array(x_data, dtype=np.float), np.array(y_data, dtype=np.float), np.array(ids), columns, np.array(weights)


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