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

    smallTrainX, smallTrainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modSmallTrain.csv",
                                                 selectFeatures=False, enc=None)

    # ---------- make pipes and params
    simple = True
    pipe_prep, params_prep = prepPipes(simple=simple)
    pipe_class, params_class = classifierPipes(simple=simple)
    pipe_reg, params_reg = regressorPipes(simple=simple)

    # pipe = Pipeline(steps=pipe_prep.steps + pipe_reg.steps)
    # allParamsDict = dict(params_prep.items() + params_reg.items())

    pipe = BinThenReg(pipe_prep, pipe_class, pipe_reg)
    params = BinThenReg.make_params_dict(params_prep, params_class, params_reg)

    # ---------- select learner
    data = DatasetPair(np.array(smallTrainX), np.array(smallTrainY))
    #
    # randomStates = [0, 1]       # try multiple random states for better calibration
    # popSize = 8
    #
    # cvObjs = [StratifiedShuffleSplit([0 if y == 0 else 1 for y in data.Y], n_iter=5, test_size=0.25,
    #                                random_state=randomState) for randomState in randomStates]
    # initPop = [[np.random.randint(len(v)) for v in params.items()] for _ in range(popSize)]
    #
    # dt = datetime.now()
    #
    # _, bestParams, score = fitClfWithGridSearch('GB', pipe, params, data,
    #                                             saveToDir='/home/jj/code/Kaggle/Loan Default Prediction/output/gridSearchOutput',
    #                                             useJJ=True, score_func=mean_absolute_error, n_jobs=20, verbosity=3,
    #                                             minimize=True, cvObjs=cvObjs, maxLearningSteps=10,
    #                                             numConvergenceSteps=3, convergenceTolerance=0, eliteProportion=0.1,
    #                                             parentsProportion=0.4, mutationProportion=0.1, mutationProbability=0.1,
    #                                             mutationStdDev=None, populationSize=popSize)
    #
    # bestPipe = clone(pipe)
    # bestPipe.set_params(**bestParams)

    bestPipe = loadObject('/home/jj/code/Kaggle/Loan Default Prediction/output/gridSearchOutput/GB.pk')['best_estimator']
    print '>>> Classifier pipe roc score:', bestPipe.classification_roc(data.X, data.Y)

    # ga = GAGridSearchCV_JJ(data=data, pipe=pipe, allParamsDict=params, cv=cvObj, minimize=True,
    #                   maxValsForInputs=[len(x)-1 for x in params.values()],
    #                   initialEvaluables=initPop[0], initialPopulation=initPop, populationSize=popSize, verbosity=5,
    #                   maxLearningSteps=10, numConvergenceSteps=5, convergenceTolerance=0, eliteProportion=0.1,
    #                   parentsProportion=0.4, mutationProportion=0.1, mutationProbability=0.1, mutationStdDev=None,
    #                   n_jobs=20, scoreFunc=mean_absolute_error, numCvFolds=4)
    # ga.learn()
    # bestParams = getParamsFromIndices(ga.bestEvaluable, params)
    # print_GSCV_info(ga, isGAJJ=True, bestParams=bestParams)
    # bestPipe = clone(pipe)
    # bestPipe.set_params(**bestParams)

    # gscv = GridSearchCV(pipe, params, loss_func=mean_absolute_error, n_jobs=20, cv=4, verbose=5)
    # gscv.fit(smallTrainX, smallTrainY)
    # print_GSCV_info(gscv)
    # bestPipe = gscv.best_estimator_
    #
    print 'CV Took', datetime.now() - dt



    # ---------- double-check cv score

    dt = datetime.now()
    print 'jj score:', quick_score(bestPipe, smallTrainX, smallTrainY)
    print 'quick jj score took', datetime.now() - dt

    # ---------- learn the full training data
    fullTrainX, fullTrainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTrain.csv",
                                               selectFeatures=False, enc=None)
    dt = datetime.now()
    bestPipe.fit(fullTrainX, fullTrainY)
    print 'Fitting to full training data took', datetime.now() - dt

    # ----------  predict & write to file
    write_predictions_to_file(predictor=bestPipe,
                              testDataFname="/home/jj/code/Kaggle/Loan Default Prediction/Data/modTest.csv",
                              enc=enc,
                              outputFname="/home/jj/code/Kaggle/Loan Default Prediction/submissions/GBR_binReg_GAJJ_notSimple.csv")

    print '----- FIN -----'