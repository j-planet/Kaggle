import sys
import numpy as np
from pprint import pprint

sys.path.extend(['/home/jj/code/Kaggle/allstate'])

from sklearn.preprocessing import Imputer, Normalizer
from sklearn.metrics import auc_score, accuracy_score, precision_score

from helpers import *
from globalVars import *
from pipes import make_pipes
from Kaggle.CV_Utilities import fitClfWithGridSearch
from Kaggle.utilities import DatasetPair, impute_field


def impute_risk_factors(name, readFromFiles=True):
    """
    normalizes and imputes X
    rounds y
    @param name: name of the training/testing input file
    @return: X_present, y_present, X_missing
    """
    inputTable = condense_data(name, DATA_DIR, isTraining=False, readFromFiles=readFromFiles,
                               outputDir=CONDENSED_TABLES_DIR)[1]

    X_present, y_present, X_missing = impute_field(inputTable, 'risk_factor')

    X_present = Normalizer().fit_transform(Imputer().fit_transform(X_present))
    y_present = y_present.round()
    X_missing = Normalizer().fit_transform(Imputer().fit_transform(X_missing))

    return X_present, y_present, X_missing


# ---------- read data ----------
X_cal, y_cal, _ = impute_risk_factors('smallerTrain')

# ---------- calibrate to find the best classifier ----------
bestScore = -1
bestPipe = None
bestParams = None

for name, (pipe, params) in make_pipes().iteritems():
    print '>'*10, name, '<'*10
    _, cur_bestParams, cur_bestScore = fitClfWithGridSearch(name + '_risk', pipe, params, DatasetPair(np.array(X_cal), y_cal),
                                                            saveToDir='/home/jj/code/Kaggle/allstate/output/gridSearchOutput',
                                                            useJJ=True, score_func=accuracy_score, n_jobs=N_JOBS, verbosity=0,
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

# ---------- train the classifier ----------
X_train, y_train, X_pred = impute_risk_factors('smallTrain')
bestPipe.fit(X_train, y_train)

# ---------- predict missing risk factors ----------
y_pred = bestPipe.predict(X_pred)


print y_pred.shape, X_pred.shape