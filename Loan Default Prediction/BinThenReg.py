from datetime import datetime
import numpy as np
from copy import deepcopy

from sklearn import clone
from sklearn.metrics import mean_absolute_error, zero_one, roc_auc_score
from sklearn.cross_validation import StratifiedShuffleSplit

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

    def classification_metrics(self, X, y, n_iter=10, test_size=0.25, random_state=0):
        """
        returns the roc auc of the classifier binary only., and the portion of correct predictions via CV
        @param y: all non-zero will be set to 1
        @param n_iter, test_size: StratifiedShuffleSplit parameters
        @param random_state: random state used for StratifiedShuffleSplit
        @return: roc, accuracy, accuracy_zero, accuracy_one
        """

        roc = 0
        accuracy = 0
        accuracy_zero = 0   # portion of zeros correctly predicted
        accuracy_one = 0    # portion of ones correctly predicted

        y = np.array([0 if d == 0 else 1 for d in y])
        prePipe = clone(self.common_preprocessing_pipe)
        pipeToUse = clone(self.classifier_pipe)
        cvObj = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=test_size,
                               random_state=random_state)

        for trainInds, testInds in cvObj:    # all cv data
            trainX = X[trainInds]
            trainY = y[trainInds]
            testX = X[testInds]
            testY = y[testInds]

            trainX = prePipe.fit_transform(trainX)
            testX = prePipe.transform(testX)
            pipeToUse.fit(trainX, trainY)
            y_scores = pipeToUse.predict_proba(testX)
            y_pred = pipeToUse.predict(testX)

            temp = next((i for i in range(len(testY)) if y_pred[i]==1), None)

            roc += roc_auc_score(testY, y_scores[:, 1])
            accuracy += sum(y_pred == testY)*1./len(testY)
            accuracy_zero += 1. * sum(np.logical_and(y_pred == testY, testY == 0)) / sum(testY == 0)
            accuracy_one += 1. * sum(np.logical_and(y_pred == testY, testY == 1)) / sum(testY == 1)

        roc /= n_iter
        accuracy_zero /= n_iter
        accuracy_one /= n_iter
        accuracy /= n_iter

        print '>>> The classifier has roc = %0.3f, zero-accuracy = %0.3f, ' \
              'one-accuracy = %0.3f, overall accuracy = %0.3f.' \
              % (roc, accuracy_zero, accuracy_one, accuracy)

        return roc, accuracy, accuracy_zero, accuracy_one

    @staticmethod
    def make_params_dict(prepParamsDict, classParamsDict, regParamsDict):
        """
        @return: a dictionary
        """

        d1 = dict(('common_preprocessing_pipe__' + k, v) for k, v in prepParamsDict.iteritems())
        d2 = dict(('classifier_pipe__' + k, v) for k, v in classParamsDict.iteritems())
        d3 = dict(('regressor_pipe__' + k, v) for k, v in regParamsDict.iteritems())

        return dict(d1.items() + d2.items() + d3.items())
