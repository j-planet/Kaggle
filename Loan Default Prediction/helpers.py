from dbus.decorators import method
import numpy as np
import pandas
import csv
from copy import copy

import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler
from sklearn.decomposition import KernelPCA, PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression, RFECV
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.base import BaseEstimator, TransformerMixin

from globalVars import *
from Kaggle.utilities import plot_histogram


def select_features(origXData, yData, pcaVarSum = 0.95):
    """
    Feature selection
    @param origXData: a pandas data frame
    @param yData: a vector
    @param pcaVarSum:
        at the PCA stage choose the number of features that explain at least this portion of the total variance
    @return: selected xData (np.array), the imputation object, the scaler object, the pca object, indices from the RF
    """

    columns = origXData.columns

    # fill in missing values first
    imp = Imputer(strategy='mean')
    xData = imp.fit_transform(origXData, yData)

    # standardize
    scaler = StandardScaler()
    xData = pandas.DataFrame(data=scaler.fit_transform(xData, yData), columns=columns)

    # decompose
    pca = PCA()
    xData = pca.fit_transform(xData, yData)
    # choose the number of features that explain at least 95% of the variance
    temp = np.cumsum(pca.explained_variance_ratio_)
    numFeatures_pca = next((i for i in range(len(temp)) if temp[i] > pcaVarSum), len(temp)-1) + 1
    xData = xData[:, range(numFeatures_pca)]
    print 'PCA reduced the number of features to:', numFeatures_pca

    # RFECV: not used for now. takes waaaaay too long
    # rfecv = RFECV(SVR(kernel='linear'), step=1, cv=5, verbose=5)
    # xData = rfecv.fit_transform(xData, yData)

    # Random Forest
    numFeatures_rf = int(xData.shape[1] * 0.5)
    forest = ExtraTreesRegressor(n_estimators=25, n_jobs=20)
    forest.fit(xData, yData)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1][:numFeatures_rf]
    xData = xData[:, indices]

    print "Random Forest reduced the number of features to:", numFeatures_rf

    # some rf plotting
    plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
    plt.show()

    return xData, imp, scaler, pca, indices


def print_missing_values_info(data):
    """
    Prints the number of missing data columns and values of a pandas data frame.
    @param data 2D pandas data frame
    @return None
    """

    temp_col = pandas.isnull(data).sum()
    temp_row = pandas.isnull(data).sum(axis=1)

    print 'The data has', (temp_col > 0).sum(), 'or', round(100. * (temp_col > 0).sum() / data.shape[1], 1), '% columns with missing values.'
    print 'The data has', (temp_row > 0).sum(), 'or', round(100. * (temp_row > 0).sum() / data.shape[0], 1), '% rows with missing values.'

    print 'The data has', temp_col.sum(), 'or', round(
        100. * temp_col.sum() / (data.shape[0] * data.shape[1]), 1), '% missing values.'


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


def make_data(dataFname, selectFeatures):
    """
    reads x and y data (no imputation, yes feature selection)
    @param dataFname: name of the training csv file
    @param selectFeatures: if True output only the best features. Otherwise, the original data. True by default.
    @return xdata, ydata (None if test data), ids
    """

    origData = pandas.read_csv(dataFname)
    ids = origData['id']

    # remove unused columns
    if 'Unnamed: 0' in origData.columns: del origData['Unnamed: 0']
    del origData['id']

    # separate into X & y values
    xData = origData[[col for col in origData.columns if not col=='loss']]
    set_vars_as_type(xData, discreteVars, object)
    yVec = origData.loss if 'loss' in origData.columns else None

    print_missing_values_info(origData)

    # feature selection
    if selectFeatures:

        columns_final = ['f536', 'f602', 'f603', 'f4', 'f605', 'f6', 'f2', 'f696', 'f473', 'f344', 'f261', 'f767', 'f285', 'f765', 'f666',
                         'f281', 'f282', 'f665', 'f221', 'f323', 'f322', 'f47', 'f5', 'f103', 'f667', 'f68', 'f67', 'f474', 'f675', 'f674',
                         'f676', 'f631', 'f462', 'f468', 'f425', 'f400', 'f778', 'f405', 'f776', 'f463', 'f428', 'f471', 'f777', 'f314', 'f211',
                         'f315', 'f252', 'f251', 'f426', 'f12', 'f11', 'f70']
        filteredXData = xData[list(columns_final)]
    else:   # use ALL features
        filteredXData = xData

    return filteredXData, yVec, ids


def write_predictions_to_file(ids, predictions, outputFname):
    """
    write output to file
    """

    featureSelectionOutput = np.transpose(np.vstack((ids, predictions.round().astype(int))))

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

    def __init__(self, method="PCA", num_components = None, total_var = None):
        """
        Constructor
        @param num_components: the number of features desired
        @param total_var: the portion of variance expllained. exactly one of num_components and total_var is None
        @param method:
            type of PCA to use. {"PCA", "KernelPCA"}. So far only PCA is supported.
        @return: nothing
        """

        assert method in ['PCA'], 'Unexpected method %s' % method
        assert (num_components is None) != (total_var is None), 'Exactly one of num_components and total_var is None.'

        if method == "PCA":
            self._pca = PCA()

        self._method = method

        if num_components is None:  # given total var
            self._fixed_num_components = False
            self._total_var = total_var
        else:
            self._fixed_num_components = True
            self._num_components = num_components

    def fit(self, X, y=None):
        """
        @return: self
        """

        self._pca.fit(X, y)

        # figure out the number of components needed, only if it's not given in the constructor
        if not self._fixed_num_components:
            temp = np.cumsum(self._pca.explained_variance_ratio_)
            self._num_components = next((i for i in range(len(temp)) if temp[i] > self._total_var), len(temp)-1) + 1

        print 'Selected %i features' % self._num_components

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


class RandomForester(BaseEstimator, TransformerMixin):

    def __init__(self, num_features, n_estimators, max_depth=None, min_samples_split=2, n_jobs=20):
        """
        Constructor
        @param num_features:
            number of features. if in (0,1), represents the proportion of features. if >1,
            represents the final number of features.
        @param n_estimators, max_depth, min_samples_split, n_jobs: params used in ExtraTreesRegressor
        """

        self._num_features = num_features

        self._forest = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_split=min_samples_split, n_jobs=n_jobs)

    def fit(self, X, y=None):
        """
        @return: self
        """

        self._forest.fit(X, y)

        return self

    def transform(self, X):
        """
        @return: new x
        """

        importances = self._forest.feature_importances_
        indices = np.argsort(importances)[::-1][:self._num_features]
        return X[:, indices]

    def fit_transform(self, X, y=None, **fit_params):
        """
        @return: new x
        """

        self.fit(X, y)

        return self.transform(X)

    def plot(self, num_features='auto'):
        """
        makes a bar plot of feature importances and corresp. standard deviations
        @param num_features:
            number of features to show.
              'auto': same as the class' number of selected features
              'all': all features
              a number: specific # features
        """

        importances = self._forest.feature_importances_


        if num_features == 'auto':
            numTicks = self._num_features
        elif num_features== 'all':
            numTicks = len(importances)
        elif isinstance(num_features, int):
            numTicks = num_features
        else:
            raise Exception('Invalid num_features provided:', num_features)

        indices = np.argsort(importances)[::-1][:numTicks]
        std = np.std([tree.feature_importances_ for tree in self._forest.estimators_], axis=0)

        plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
        plt.show()