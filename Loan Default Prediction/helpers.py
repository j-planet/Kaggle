from dbus.decorators import method
import numpy as np
import pandas
import csv
from copy import copy, deepcopy
from pprint import pprint
from datetime import datetime

import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import KernelPCA, PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression, RFECV
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error
from tornado.test import import_test

from globalVars import *
from Kaggle.utilities import plot_histogram, jjcross_val_score, print_missing_values_info


def set_vars_as_type(df, varNames, dtype):
    """
    Set certain variables of a pandas df to a pre-defined data type. Changes df in place.
    @param df pandas data frame
    @param varNames list of strings
    @param dtype data type
    """

    myVars = list(set(df.columns).intersection(set(varNames)))
    df[myVars] = df[myVars].astype(dtype)


def impute_data(xData, yData):
    """
    Filles in missing values
    @param xData: a pandas data frame of x values
    @param yData: a vector of y values
    @return: new xdata (a pandas df)
        the imputer that can be called to transform future data
    """

    imp = Imputer(strategy='mean')
    imp.fit(xData, yData)
    newXData = pandas.DataFrame(imp.transform(xData), columns=xData.columns)

    return newXData, imp


def make_data(dataFname, enc, features=None):
    """
    reads x and y data (no imputation, yes feature selection)
    also encodes the categorical features f776 and f777
    @param dataFname: name of the training csv file
    @param features: specific features to use. None by default.
    @param enc: the OneHotEncoder. None for training data, not-None for testing data
    @return xdata, ydata (None if test data), ids, enc (OneHotEncoder for f776 and f777)
    """

    origData = pandas.read_csv(dataFname)
    ids = origData['id']

    # remove unused columns
    if 'Unnamed: 0' in origData.columns: del origData['Unnamed: 0']
    del origData['id']

    # remove "data leakage" columns
    for f in prohobitedFeatures:
        del origData[f]

    # separate into X & y values
    xData = origData[[col for col in origData.columns if not col=='loss']]
    set_vars_as_type(xData, discreteVars, object)
    yVec = origData.loss if 'loss' in origData.columns else None

    # try f528 - f274
    xData['f528f274'] = xData['f528'] - xData['f274']

    # encode the categorical features f776 and f777
    if enc is None:
        enc = OneHotEncoder(n_values=[2, 2])
        enc.fit(xData[['f776', 'f777']])

    xData[['f776_isZero', 'f776_isOne', 'f777_isZero', 'f777_isOne']] = pandas.DataFrame(enc.transform(xData[['f776', 'f777']]).toarray())
    del xData['f776']
    del xData['f777']

    print_missing_values_info(origData)

    # feature selection
    if features:
        filteredXData = xData[features]
    else:   # use ALL features
        filteredXData = xData

    return filteredXData, yVec, ids, enc


def write_predictions_to_file(predictor, testDataFname, enc, outputFname, features=None):
    """
    write output to file
    """

    testData, _, testDataIds, _ = make_data(testDataFname, features=features, enc=enc)

    dt = datetime.now()
    predictions = predictor.predict(testData)
    print 'predicting took', datetime.now() - dt

    featureSelectionOutput = np.transpose(np.vstack((testDataIds, predictions.round().astype(int))))

    with open(outputFname, 'wb') as outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(['id', 'loss'])
        writer.writerows(featureSelectionOutput)


def split_class_reg(xData, yData):
    """
    transforms a regression problem into a classification (0 vs >0) problem first
    @param yData: non-negative integers
    @return: binarilized y data (0s and 1s), xData where y>0, yData where y>0, mask for the rows where y>0
    """

    nonZeroMask = yData > 0

    binaryY = copy(yData)
    binaryY[nonZeroMask] = 1

    return binaryY, xData[nonZeroMask], yData[nonZeroMask], nonZeroMask


class PcaEr(BaseEstimator, TransformerMixin):
    """
    class used for dimension reduction via PCA
    """

    def __init__(self, method="PCA", fixed_num_components=None, total_var=None, whiten=False):
        """
        Constructor
        @param fixed_num_components: the number of features desired
        @param total_var: the portion of variance expllained. exactly one of num_components and total_var is None
        @param method:
            type of PCA to use. {"PCA", "KernelPCA"}. So far only PCA is supported.
        @param whiten: to be used in PCA, whether to remove correlations in the input
        @return: nothing
        """

        assert method in ['PCA'], 'Unexpected method %s' % method
        assert (fixed_num_components is None) != (total_var is None), \
            'Exactly one of fixed_num_components and total_var is None.'

        self.whiten = whiten

        if method == "PCA":
            self._pca = PCA(whiten=self.whiten)

        self.method = method

        if fixed_num_components is None:  # given total var
            self._fixed_num_components = False
            self.total_var = total_var
        else:
            self._fixed_num_components = True
            self._num_components = fixed_num_components

    def fit(self, X, y=None):
        """
        @return: self
        """

        self._pca.fit(X, y)

        # figure out the number of components needed, only if it's not given in the constructor
        if not self._fixed_num_components:
            temp = np.cumsum(self._pca.explained_variance_ratio_)
            self._num_components = next((i for i in range(len(temp)) if temp[i] > self.total_var), len(temp)-1) + 1

        # print '%s with %f total var, selected %i features' % (self.method, self.total_var, self._num_components)

        return self

    def transform(self, X):
        """
        @return: new x
        """

        return self._pca.transform(X)[:, range(self._num_components)]

    def fit_transform(self, X, y=None, **fit_params):
        """
        @return: new x
        """

        self.fit(X, y)
        return self.transform(X)




def quick_score(clf, X, y, cv=5, n_jobs=20):
    """ returns the cv score of a classifier
    """

    return jjcross_val_score(clf, X, y, mean_absolute_error, cv, n_jobs=n_jobs).mean()


