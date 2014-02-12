from datetime import datetime
import numpy as np
from copy import deepcopy

from sklearn.ensemble.tests.test_gradient_boosting import test_check_max_features
from sklearn.metrics import mean_absolute_error, zero_one
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

from Kaggle.utilities import makePipe, Normalizer
from helpers import *
from pipes import *


class BinThenReg(BaseEstimator, TransformerMixin):
    """
    first classify target values as 0 or positive, then predict the positive values
    """

    def __init__(self, common_preprocessing_pipe, classifier_pipe, regressor_pipe):
        """
        @param common_preprocessing_pipe: the preprocessing step done prior to BOTH classifying and regressing
        """

        # self.common_preprocessing_pipe = deepcopy(common_preprocessing_pipe)
        # self.classifier_pipe = deepcopy(classifier_pipe)
        # self.regressor_pipe = deepcopy(regressor_pipe)
        self.common_preprocessing_pipe = common_preprocessing_pipe
        self.classifier_pipe = classifier_pipe
        self.regressor_pipe = regressor_pipe

    def set_params(self, **params):

        self.common_preprocessing_pipe.set_params(**dict((name[(len('common_preprocessing_pipe'))+2:], v)
                                                         for name, v in params.iteritems()
                                                         if 'common_preprocessing_pipe' in name))

        self.classifier_pipe.set_params(**dict((name[(len('classifier_pipe'))+2:], v)
                                               for name, v in params.iteritems()
                                               if 'classifier_pipe' in name))

        self.regressor_pipe.set_params(**dict((name[(len('regressor_pipe'))+2:], v)
                                              for name, v in params.iteritems()
                                              if 'regressor_pipe' in name))

        return self

    def fit(self, X, y):

        binaryY, _, regY, nonZeroMask = split_class_reg(X, y)

        # apply common_preprocessing_pipe
        newX = self.common_preprocessing_pipe.fit_transform(X)

        # apply classifier_pipe
        self.classifier_pipe.fit(newX, binaryY)

        # apply regressor_pipe
        self.regressor_pipe.fit(newX[nonZeroMask], regY)

    def predict(self, X):

        # apply common_preprocessing_pipe
        newX = self.common_preprocessing_pipe.fit_transform(X)  # just transform?

        # apply classifier_pipe
        binaryOutput = self.classifier_pipe.predict(newX)
        nonZeroMask = binaryOutput > 0

        # apply regressor_pipe
        if sum(nonZeroMask) > 0:
            regOutput = self.regressor_pipe.predict(newX[nonZeroMask])
            regOutput[(regOutput>0) & (regOutput<1)] = 1            # set 0.sth to 1
            regOutput = regOutput.round().astype(int)

            # combine binary and regression output
            binaryOutput[nonZeroMask] = regOutput
        else:
            print 'All zero output...'

        return binaryOutput

    @staticmethod
    def make_params_dict(prepParamsDict, classParamsDict, regParamsDict):
        """
        @return: a dictionary
        """

        d1 = dict(('common_preprocessing_pipe__' + k, v) for k, v in prepParamsDict.iteritems())
        d2 = dict(('classifier_pipe__' + k, v) for k, v in classParamsDict.iteritems())
        d3 = dict(('regressor_pipe__' + k, v) for k, v in regParamsDict.iteritems())

        return dict(d1.items() + d2.items() + d3.items())
