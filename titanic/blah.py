__author__ = 'jjin'
import sys, os
sys.path.append('H:/ff/Kaggle')
sys.path.append('C:/code/Kaggle')

from utilities import DatasetPair, MyPool, jjcross_val_score, makePipe, printDoneTime, MissingValueFiller, Normalizer, loadObject
from titanicutilities import buildModel, fitClfWithGridSearch, readData
from globalVariables import fillertoTry, normalizerToTry, classifiersToTry, rootdir, svc_f, svc_m, rf_f, rf_m
from GradientBoost_JJ import GradientBoost_JJ
from GA_JJ import getParamsFromIndices

from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.ensemble.gradient_boosting import LeastSquaresError
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import numpy as np
from pprint import pprint
import multiprocessing
from random import randint
from time import time


def init(*args):
    global a,b
    a,b = args


def sumjj(args):
    global a,b
    l = args
    return l + a + b


def work3(n):
    print 'WORK3 with n =', n
    return n


def work2(num_procs):

    print("WORK2: Creating %i (daemon) workers and jobs in child." % num_procs)

    pool = multiprocessing.Pool(num_procs)
    result = pool.map(work3, np.repeat(num_procs, num_procs))
    pool.close()
    pool.join()

    return result


def work1(num_procs):
    print("WORK1: Creating %i (daemon) workers and jobs in child." % num_procs)
    # pool = multiprocessing.Pool(num_procs)
    pool = MyPool(num_procs)

    result = pool.map(work2,
                      np.repeat(num_procs, num_procs))

    # The following is not really needed, since the (daemon) workers of the
    # child's pool are killed when the child is terminated, but it's good
    # practice to cleanup after ourselves anyway.
    pool.close()
    pool.join()
    return result


def test():
    print("Creating 5 (non-daemon) workers and jobs in main process.")
    pool = MyPool(5)

    result = pool.map(work1, np.arange(5)+1)

    pool.close()
    pool.join()
    print(result)


def func1(**kwargs):
    print 'in func 1'
    func2(**kwargs)

def func2(a):
    print 'in func 2'
    print 'a =', a

if __name__ == '__main__':

    data, testData, fieldMaps, _, _ = readData(outputDir=rootdir)
    random_states = range(10)
    useJJ = True

    # make pipe and allParamsDict
    # name = 'svc'
    # step1 = fillertoTry
    # step2 = normalizerToTry
    # step3 = (name,classifiersToTry[name])
    # pipe, allParamsDict = makePipe([step1, step2, step3])

    # pprint(allParamsDict)
    # print '>'*20
    # p = loadObject("H:/allparamsdict")
    # pprint(getParamsFromIndices([2, 2, 1, 0, 2], p))

    # trainX, trainY, testX, testY = loadObject("H:/allcvdata")[1]
    # params = getParamsFromIndices([2, 2, 1, 0, 2], allParamsDict)
    # pipe.set_params(**params)
    # pipe.fit(trainX, trainY)
    # print 'here'
    # print accuracy_score(testY, pipe.predict(testX))

    t0 = time()

    # newpipe, best_params, score = fitClfWithGridSearch(name, pipe, allParamsDict, data, os.path.join(rootdir, 'intermediate results'),
    #                                                    n_jobs=20, cvSplitNum=10, random_state=random_state, useJJ=useJJ, verbosity=2,
    #
    #                                                    maxLearningSteps=30, numConvergenceSteps=4, eliteProportion=0.1, saveCache=True,
    #                                                    parentsProportion=0.4, populationSize=15,
    #                                                    mutationProbability=0.3, mutationProportion=0.2, mutationStdDev=None, maxDuplicateProportion=0)
    # print 'SCORE =', score

    # print 'x'*50
    buildModel(data, testData, fieldMaps, selectedClfs=['svc'], useJJ=useJJ, n_jobs=20, writeResults=True,
              colNames=['sex', 'name', 'embarked'], cvNumSplits=10, random_states=random_states, verbose=True,

              maxLearningSteps=30, numConvergenceSteps=5, eliteProportion=0.1, saveCache=True,
              parentsProportion=0.4, populationSize=12, verbosity=2,
              mutationProbability=0.3, mutationProportion=0.2, mutationStdDev=None, maxDuplicateProportion=0)


    printDoneTime(t0)
    print '>>>> FIN <<<<'

