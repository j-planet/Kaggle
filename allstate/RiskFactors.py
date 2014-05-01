import sys
import numpy as np
from pprint import pprint

sys.path.extend(['/home/jj/code/Kaggle/allstate'])

from sklearn.preprocessing import Imputer, Normalizer
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

from helpers import *
from globalVars import *
from pipes import make_pipes
from Kaggle.CV_Utilities import fitClfWithGridSearch
from Kaggle.utilities import DatasetPair, impute_field


def impute_risk_factors(inputTable):
    """
    normalizes and imputes X
    rounds y
    @param name: name of the training/testing input file
    @return: X_present, y_present, X_missing
    """

    X_present, y_present, X_missing, ind_missing = impute_field(inputTable, 'risk_factor')

    X_present = Normalizer().fit_transform(Imputer().fit_transform(X_present))
    y_present = y_present.round()
    X_missing = Normalizer().fit_transform(Imputer().fit_transform(X_missing))

    return X_present, y_present, X_missing, ind_missing


class ImputerJJ(BaseEstimator, TransformerMixin):
    """
    class used for imputation of features with missing values
    """

    def __init__(self, calibrationTable, score_func=accuracy_score):
        """
        calibrate a classifier
        @param calibrationTable: a pandas data frame
        """

        print '--------- Calibrating Imputer -----------'
        X_cal, y_cal, _, _ = impute_risk_factors(calibrationTable)

        bestScore = -1
        bestPipe = None
        bestParams = None

        for name, (pipe, params) in make_pipes().iteritems():
            print '>'*10, name, '<'*10
            _, cur_bestParams, cur_bestScore = fitClfWithGridSearch(name + '_risk', pipe, params, DatasetPair(np.array(X_cal), y_cal),
                                                                    saveToDir='/home/jj/code/Kaggle/allstate/output/gridSearchOutput',
                                                                    useJJ=True, score_func=score_func, n_jobs=N_JOBS, verbosity=0,
                                                                    minimize=False, cvSplitNum=5,
                                                                    maxLearningSteps=10,
                                                                    numConvergenceSteps=4, convergenceTolerance=0, eliteProportion=0.1,
                                                                    parentsProportion=0.4, mutationProportion=0.1, mutationProbability=0.1,
                                                                    mutationStdDev=None, populationSize=6)

            if cur_bestScore > bestScore:

                bestScore = cur_bestScore
                bestPipe = clone(pipe)
                bestPipe.set_params(**cur_bestParams)
                bestParams = cur_bestParams

        print '----> best score:', bestScore
        pprint(bestParams)

        self._imputer = bestPipe

    def fit(self, trainTable):
        """
        train the calibrated classifier
        @param trainTable: a pandas data frame
        @return: self
        """

        X_train, y_train, _, _ = impute_risk_factors(trainTable)
        self._imputer.fit(X_train, y_train)

        return self

    def transform(self, predTable):
        """
        predict missing values using the classifier
        alteras predTable
        @return: the altered predTable
        """

        _, _, X_pred, ind_missing = impute_risk_factors(predTable)
        y_pred = self._imputer.predict(X_pred)

        predTable['risk_factor'][ind_missing] = y_pred

        return predTable

    def fit_transform(self, X, y=None, **fit_params):
        """
        train the calibrated classifier and use it to predict missing values
        @param X: is in fact trainPredTable
        """

        trainPredTable = X
        self.fit(trainPredTable)
        return self.transform(trainPredTable)


if __name__ == '__main__':
    imp = ImputerJJ(condense_data('tinyTrain', isTraining=True, readFromFiles=True)[1])
    imp.fit(condense_data('tinyTrain', isTraining=True, readFromFiles=True)[1])


    predTable = condense_data('tinyTrain', isTraining=True, readFromFiles=True)[1]
    print predTable.risk_factor
    imp.transform(predTable)

    print predTable.risk_factor
