from datetime import datetime
from pprint import pprint
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import clone

from helpers import *
from globalVars import *
from pipes import *
from BinThenReg import *
from Kaggle.utilities import DatasetPair
from Kaggle.GA_JJ import GAGridSearchCV_JJ, getParamsFromIndices


if __name__=='__main__':

    smallTrainX, smallTrainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modSmallTrain.csv",
                                                 selectFeatures=False, enc=None)
    fullTrainX, fullTrainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTrain.csv",
                                               selectFeatures=False, enc=None)

    # ---------- make pipes and params
    pipe_prep, params_prep = prepPipes(simple=True)
    pipe_class, params_class = classifierPipes(simple=True)
    pipe_reg, params_reg = regressorPipes(simple=True)

    # pipe = Pipeline(steps=pipe_prep.steps + pipe_reg.steps)
    # allParamsDict = dict(params_prep.items() + params_reg.items())

    pipe = BinThenReg(pipe_prep, pipe_class, pipe_reg)
    params = BinThenReg.make_params_dict(params_prep, params_class, params_reg)

    # ---------- select learner
    data = DatasetPair(fullTrainX, fullTrainY)

    randomState = 0
    numCvFolds = 4
    popSize = 10
    cvObj = StratifiedShuffleSplit(data.Y, numCvFolds, test_size=0.25, random_state=randomState)
    initPop = [dict((k, v[np.random.randint(len(v))]) for k, v in params.iteritems()) for _ in range(popSize)]

    dt = datetime.now()

    ga = GAGridSearchCV_JJ(data=data, pipe=pipe, allParamsDict=params, cvs=cvObj, minimize=true,
                      maxValsForInputs=[len(x)-1 for x in params.values()],
                      initialEvaluables=initPop[0], initialPopulation=initPop, populationSize=popSize, verbosity=5,
                      maxLearningSteps=10, numConvergenceSteps=5, convergenceTolerance=0, eliteProportion=0.1,
                      parentsProportion=0.4, mutationProportion=0.1, mutationProbability=0.1, mutationStdDev=None,
                      n_jobs=20, scoreFunc=mean_absolute_error)
    ga.learn()
    bestParams = getParamsFromIndices(ga.bestEvaluable, params)
    print_GSCV_info(ga, isGAJJ=True, bestParams=bestParams)
    bestPipe = clone(pipe)
    bestPipe.set_params(**bestParams)

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
    dt = datetime.now()
    bestPipe.fit(fullTrainX, fullTrainY)
    print 'Fitting to full training data took', datetime.now() - dt

    # ----------  predict & write to file
    write_predictions_to_file(predictor=bestPipe,
                              testDataFname="/home/jj/code/Kaggle/Loan Default Prediction/Data/modTest.csv",
                              enc=enc,
                              outputFname="/home/jj/code/Kaggle/Loan Default Prediction/submissions/GBR_binReg_GAJJ.csv")

    print '----- FIN -----'