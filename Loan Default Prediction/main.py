from datetime import datetime
from pprint import pprint
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit, StratifiedShuffleSplit
from sklearn import clone

from helpers import *
from globalVars import *
from pipes import *
from BinThenReg import *
from Kaggle.utilities import DatasetPair, print_GSCV_info
from Kaggle.CV_Utilities import fitClfWithGridSearch, loadObject


if __name__=='__main__':

    # featuresToTry = [u'f528f274', u'f674', u'f776_isOne', u'f776_isZero', u'f222', u'f221', u'f2', u'f653', u'f767',
    #                  u'f251', u'f25', u'f670', u'f675', u'f293', u'f471', u'f768', u'f766', u'f324', u'f377', u'f73',
    #                  u'f318', u'f329', u'f592', u'f676', u'f326']
    featuresToTry = None
    smallTrainX, smallTrainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modSmallTrain.csv",
                                                 enc=None, features=featuresToTry)
    fullTrainX, fullTrainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTrain.csv",
                                               enc=None, features=featuresToTry)

    # ---------- make pipes and params
    simple = True
    name = 'logistic'
    useSmallForCalibration = False
    pipe_prep, params_prep = prepPipes(simple=simple)
    pipe_class, params_class = classifierPipes(simple=simple, name=name, usePCA=False, useRF=True)
    pipe_reg, params_reg = regressorPipes(simple=simple, usePCA=False, useRF=True)
    pipe = BinThenReg(pipe_prep, pipe_class, pipe_reg)
    params = BinThenReg.make_params_dict(params_prep, params_class, params_reg)

    # ---------- select learner
    if useSmallForCalibration:
        trainX = np.array(smallTrainX)
        trainY = np.array(smallTrainY)
    else:
        trainX = np.array(fullTrainX)
        trainY = np.array(fullTrainY)
    data = DatasetPair(trainX, trainY)

    randomStates = [0, 1]       # try multiple random states for better calibration
    popSize = 8

    cvObjs = [StratifiedShuffleSplit([0 if y == 0 else 1 for y in data.Y], n_iter=5, test_size=0.25,
                                   random_state=randomState) for randomState in randomStates]
    initPop = [[np.random.randint(len(v)) for v in params.items()] for _ in range(popSize)]

    dt = datetime.now()

    _, bestParams, score = fitClfWithGridSearch(name, pipe, params, data,
                                                saveToDir='/home/jj/code/Kaggle/Loan Default Prediction/output/gridSearchOutput',
                                                useJJ=True, score_func=mean_absolute_error, n_jobs=20, verbosity=3,
                                                minimize=True, cvObjs=cvObjs, maxLearningSteps=10,
                                                numConvergenceSteps=3, convergenceTolerance=0, eliteProportion=0.1,
                                                parentsProportion=0.4, mutationProportion=0.1, mutationProbability=0.1,
                                                mutationStdDev=None, populationSize=popSize)

    bestPipe = clone(pipe)
    bestPipe.set_params(**bestParams)

    print 'CV Took', datetime.now() - dt

    # bestPipe = loadObject('/home/jj/code/Kaggle/Loan Default Prediction/output/gridSearchOutput/logistic.pk')['best_estimator']

    bestPipe.classification_metrics(trainX, trainY, n_iter=10)
    # ---------- learn the full training data
    dt = datetime.now()
    bestPipe.fit(fullTrainX, fullTrainY)
    print 'Fitting to full training data took', datetime.now() - dt

    # ----------  predict & write to file
    write_predictions_to_file(predictor=bestPipe,
                              testDataFname="/home/jj/code/Kaggle/Loan Default Prediction/Data/modTest.csv",
                              enc=enc,
                              outputFname="/home/jj/code/Kaggle/Loan Default Prediction/submissions/"
                                          + name + "_noPCA_" + ("Simple" if simple else "NonSimple")
                                          + "_top25fts_fullCal.csv",
                              features=featuresToTry)

    print '----- FIN -----'