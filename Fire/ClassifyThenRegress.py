import numpy as np
from copy import copy

from sklearn.base import BaseEstimator


class ClassifyThenRegress(BaseEstimator):
    """
    First use classification to figure out which y's are positive.
    Then use regression to figure out the actual values of the positive y's.
    """

    def __init__(self, classifier, regressor, classFields, regFields):
        """
        :param classifier:
        :param regressor:
        :param classFields: array of column indicators used for classification
        :param regFields: array of column indicators used for regression
        :return:
        """

        self.classifier = classifier
        self.regressor = regressor
        self.classFields = classFields
        self.regFields = regFields

    def fit(self, X, y, sample_weight=None):
        """
        :param X: 2D numpy array
        :param y: 1D numpy vector
        :return: self
        """

        y = np.ravel(y)
        posYInd = y > 0

        # classify
        binaryY = copy(y)
        binaryY[posYInd] = 1

        x_classify = X[:, self.classFields]
        y_classify = binaryY

        if sample_weight is not None and 'sample_weight' in self.classifier.fit.func_code.co_varnames:
            self.classifier.fit(x_classify, y_classify, sample_weight=sample_weight)
        else:
            self.classifier.fit(x_classify, y_classify)

        # regress
        x_reg = X[posYInd, :][:, self.regFields]
        y_reg = y[posYInd]

        if sample_weight is not None and 'sample_weight' in self.regressor.fit.func_code.co_varnames:
            self.regressor.fit(x_reg, y_reg, sample_weight=sample_weight[posYInd])
        else:
            self.regressor.fit(x_reg, y_reg)

        return self

    def predict(self, X):
        """
        :param X: 2D numpy array
        :return: array of predictions
        """

        x_classify = X[:, self.classFields]

        # classify
        binaryY = self.classifier.predict(x_classify)  # gives 0 and 1 y's

        # regress
        posYInd = binaryY > 0
        x_reg = X[posYInd, :][:, self.regFields]
        y_reg = self.regressor.predict(x_reg)

        # combine results
        binaryY[posYInd] = y_reg

        return binaryY