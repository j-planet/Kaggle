from __builtin__ import hasattr
from _ast import Assert
import numpy as np
from pprint import pprint

from sklearn.base import BaseEstimator, clone


class GroupThenRegress(BaseEstimator):
    """
    groups the data according to a certain feature (defined in the constructor), then regress on the rest of the features
    """

    def __init__(self, pivotColInd, regressor, verbose=0): # TODO: different regressor for different categories (maybe unnecessary)
        """
        :param pivotColInd: column ind of the pivot column, must be less than X's number of columns
        :return:
        """

        self.pivotColInd = pivotColInd
        self.regressor = regressor
        self.useWeights = 'sample_weight' in self.regressor.fit.func_code.co_varnames
        self.regressors = {}
        self.overallRegressor = clone(regressor)
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        if self.verbose >= 1: print '>>>>> fitting by category'

        for val in np.unique(X[:, self.pivotColInd]):

            curClf = clone(self.regressor)
            curInd = X[:, self.pivotColInd]==val
            curX = X[curInd, :]
            curY = y[curInd]

            if self.verbose >= 2: print '\n-----', val, curX.shape, len(curY)

            if self.useWeights:
                curWeights = sample_weight[curInd]
                curClf.fit(curX, curY, sample_weight=curWeights)
            else:
                curClf.fit(curX, curY)

            self.regressors[val] = curClf

            if self.verbose >= 1:
                if hasattr(curClf, 'intercept_'):
                    print 'intercept =', curClf.intercept_

                if hasattr(curClf, 'coef_'):
                    pprint(dict(zip(range(X.shape[1]), curClf.coef_)))

        if self.verbose >= 1: print '>>>>> fitting everything'
        if self.useWeights:
            self.overallRegressor.fit(X, y, sample_weight=sample_weight)
        else:
            self.overallRegressor.fit(X, y)

        return self

    def predict(self, X):
        if self.verbose >= 1: print '>>>>> predicting'
        res = np.repeat(np.nan, X.shape[0])

        for val in np.unique(X[:, self.pivotColInd]):
            curInd = X[:, self.pivotColInd]==val
            curX = X[curInd, :]

            if self.verbose >= 2: print '\n-----', val, curX.shape

            if val in self.regressors.keys():
                if self.verbose >= 1: print val, 'value seen before.'
                clf = self.regressors[val]
            else:
                if self.verbose >= 1: print val, 'value NOT seen. using overall regressor'
                clf = self.overallRegressor

            curY = clf.predict(curX)

            # combine
            res[curInd] = curY

        assert np.isnan(res).sum()==0, 'Not all values are filled. :('

        return res