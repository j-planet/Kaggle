__author__ = 'jennyyuejin'

import os
from copy import deepcopy
from datetime import datetime
from time import time
from pprint import pprint
from collections import Iterable
from scipy.stats import mode

import numpy as np
from sklearn import clone
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedShuffleSplit, LeavePOut
from sklearn.grid_search import GridSearchCV

from Kaggle.utilities import csv2dict, getCol, integerizeList, printDoneTime, loadObject, DatasetPair, \
    jjcross_val_score, mask2DArrayByCol, makePipe, reverseDict, saveObject, getNumCvFolds
import Kaggle.GA_JJ
from globalVariables import fillertoTry, normalizerToTry, classifiersToTry, rootdir


def evaluate(y_pred, y_true):
    """ evaluates the criterium: score = % predicted correctly
    """
    return accuracy_score(y_true, y_pred)

def readData(outputDir = None, trainDataFname = None, testDataFname = None):
    """ Read and transform training and testing data
    @return: data, testData, fieldMaps, sampleWeights, testSampleWeights
    @rtype: DatasetPair, DatasetPair, dict, Iterable, Iterable
    """

    assert outputDir is not None or (trainDataFname is not None and testDataFname is not None), \
        'Cannot figure out where the data are.'

    if trainDataFname is None: trainDataFname = os.path.join(outputDir, 'data', 'train_mod.csv')
    if testDataFname is None: testDataFname = os.path.join(outputDir, 'data', 'test_mod.csv')

    data, fieldMaps, sampleWeights = readTrainingData(trainDataFname, testDataFname)
    testData, testSampleWeights = readTestingData(testDataFname, fieldMaps)

    return data, testData, fieldMaps, sampleWeights, testSampleWeights

def readTrainingData(trainDataFname, testDataFname):
    """
    @param trainDataFname: name of the training data file
    @param testDataFname: name of the testing data file. used here only to get the string->index mapping of the text columns
    @return: allDataPair, fieldMaps, sampleWeights
    """

    fieldNames = ['survived', 'pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked','weight']
    nameMap = dict(zip(fieldNames, range(len(fieldNames))))
    dataTypes = np.array([np.int, np.int, '|S82', '|S82', np.float, np.int, np.int, '|S82', np.float, '|S82', '|S82', np.int])
    outputFieldNames = []

    # ------ read original data ---------
    data = map(list, csv2dict(trainDataFname, hasHeader=True, dataTypes=dataTypes, colIndices=None, defaultNumValue=float('nan')))
    all_y = np.array(getCol(data, [nameMap['survived']]))
    all_x = list()

    # sample weights
    sampleWeights = np.array(getCol(data, [nameMap['weight']]))

    # attach numerical fields first
    for name in ['pclass', 'age', 'sibsp', 'parch', 'fare']:
        outputFieldNames.append(name)
        all_x.append(getCol(data, [nameMap[name]]))

    # attach text fields
    testData = csv2dict(testDataFname, hasHeader=True)
    numTrainingDataPts = len(data)
    fieldMaps = {}
    for name in ['sex', 'name', 'ticket', 'cabin', 'embarked']:
        outputFieldNames.append(name)
        col, fieldMap = integerizeList(getCol(data, [nameMap[name]]) + list(testData[:, nameMap[name]-1]))
        all_x.append(col[:numTrainingDataPts])
        fieldMaps[name] = fieldMap

    # normalize data
    allDataPair = DatasetPair(np.array(zip(*all_x)), all_y, outputFieldNames)

    return allDataPair, fieldMaps, sampleWeights

def readTestingData(testDataFname, fieldMaps):
    """
    read test data
    @param testDataFname: name of the testing data file
    @return: (DatasetPair with just X, sampleWeights)
    """

    fieldNames = ['pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked','weight']
    nameMap = dict(zip(fieldNames, range(len(fieldNames))))
    dataTypes = np.array([np.int, '|S82', '|S82', np.float, np.int, np.int, '|S82', np.float, '|S82', '|S82', np.int])
    outputFieldNames = []

    data = map(list, csv2dict(testDataFname, hasHeader=True, dataTypes=dataTypes, colIndices=None, defaultNumValue=float('nan')))
    all_x = list()

    # sample weights
    sampleWeights = np.array(getCol(data, [nameMap['weight']]))

    # attach numerical fields first
    for name in ['pclass', 'age', 'sibsp', 'parch', 'fare']:
        outputFieldNames.append(name)
        all_x.append(getCol(data, [nameMap[name]]))

    # attach text fields
    for name in ['sex', 'name', 'ticket', 'cabin', 'embarked']:
        outputFieldNames.append(name)
        fieldMap = fieldMaps[name]
        all_x.append([fieldMap[v] for v in getCol(data, [nameMap[name]])])

    all_x = np.array(zip(*all_x))

    return DatasetPair(all_x, fieldNames=outputFieldNames), sampleWeights


def outputTestingResultsToFile(results, outputFname):
    """
    writes testing results to file as a single column
    @param results: a vector of results
    @param outputFname: the new file, with no header
    @return: nothing
    """

    outputFile = open(outputFname, 'w')
    outputFile.writelines([str(r)+'\n' for r in results])
    outputFile.close()


def compareResultsToTrueResults(testResults, trueResFname):
    """
    compares results to true results
    @param testResults: vector of test results
    @param trueResFname: name of the file containing a column of true results
    @return: the evaluation score
    """

    y_true = getCol(csv2dict(trueResFname, hasHeader=True, dataTypes=[np.int]), 0)
    return evaluate(testResults, y_true)


def fitClfWithGridSearch(name, origpipe, paramDict, data, saveToDir, useJJ,
                         score_func = accuracy_score, test_size=0.25, n_jobs=1, cvSplitNum=10, random_states = [None],
                         overwriteSavedResult=True, verbosity=1, **fitArgs):
    """
    tries given parameters for a classifer using GridSearchCV. returns and saves the result to file.
    does NOT change the original pipe input
    @type data DatasetPair
    @param name: name of the classifier
    @param origpipe: the pipe
    @param paramDict: parameters to try
    @param saveToDir: where the intermediate result is saved
    @param test_size: portion of test data in CV (default 0.25)
    @param n_jobs: default 1
    @param overwriteSavedResult: if false, always reads from (if) existing file; if true, overwrite only when the score is better
    @return: (the pipe or the pipe's best_estimator_, parameters of the best estimator)
    """

    print '-'*5, 'Fitting', name, '-'*5
    outputFname = os.path.join(saveToDir, name + '.pk')
    pipe = deepcopy(origpipe)

    cvObjs = []
    for randomState in random_states:
        try:
            cvObjs.append(StratifiedShuffleSplit(data.Y, cvSplitNum, test_size=test_size, random_state=randomState))
        except:
            size = test_size if isinstance(test_size, int) or test_size>1 else int(round(data.dataCount * test_size))
            obj = LeavePOut(data.dataCount, size)
            res = []

            for i in range(cvSplitNum / len(obj) + 1):
                for a, b in obj:
                    res.append((a,b))
            cvObjs.append(res[:cvSplitNum])


    if not overwriteSavedResult and os.path.exists(outputFname):    # file already exists
        print name, "already exists. Reading from file."
        res = loadObject(outputFname)
        pipe = res['best_estimator']
        best_params = res['best_params']
        score = res['score']

    else:
        best_params = None
        best_estimator = None
        score = None

        # fit the pipe
        t0 = time()

        if useJJ:       # use GAGridSearchCV_JJ
            maxValues = [len(x)-1 for x in paramDict.values()]
            maxPopSize = np.prod([v+1 for v in maxValues])
            print 'maxPopSize =', maxPopSize
            popSize = min(fitArgs['populationSize'], maxPopSize)
            initPop = GA_JJ.generateInputs(maxValues, count=popSize)

            print '---------> initial population:'
            pprint(initPop)

            with GA_JJ.GAGridSearchCV_JJ(data=data, pipe=pipe, allParamsDict=paramDict, cvs=cvObjs, minimize=False, maxValsForInputs=maxValues,
                                         initialEvaluables=initPop[0], initialPopulation=initPop, n_jobs=n_jobs,
                                         **fitArgs) \
            as ga:
                ga.learn()

                score = ga.bestEvaluation
                best_params = ga.bestParams
                best_estimator = clone(pipe)
                best_estimator.set_params(**best_params)


        else:           # use GridSearchCV
            score = 0
            for cvObj in cvObjs:
                pipe = GridSearchCV(pipe, paramDict, score_func=accuracy_score, n_jobs=n_jobs, cv=cvObj)

                pipe.fit(*data.getPair())
                best_params = pipe.best_params_
                best_estimator = pipe.best_estimator_

                # print the parameters grid
                if verbosity>=2: pprint([(r[0],r[1]) for r in pipe.grid_scores_])

                score += jjcross_val_score(best_estimator, data.X, data.Y, score_func=score_func, n_jobs=n_jobs, cv=cvObj)

            score /= sum(getNumCvFolds(cv) for cv in cvObjs)

        printDoneTime(t0)

        print 'CV score:', score

        # see if it does better than before
        if os.path.exists(outputFname):
            bestScoreSoFar = loadObject(outputFname)['score']
            if verbosity>=1:
                print 'Compared to best score so far:', bestScoreSoFar
                print 'Compared to best score so far:', bestScoreSoFar

            if score > bestScoreSoFar:
                if verbosity>=1: print 'Beat best score so far for', name, '!! Saving to file...'
                saveObject({'best_estimator': best_estimator, 'best_params': best_params, 'score':score}, fname = outputFname)
            else:
                if verbosity>=1:
                    print 'Does not beat best score so far for', name
        else:
            print 'No previous scores available. Saving...'
            saveObject({'best_estimator': best_estimator, 'best_params': best_params, 'score':score}, fname = outputFname)

    print 'Best params:'
    pprint(best_params)
    return pipe, best_params, score

def fitClassifiers(trainData, useJJ, n_jobs=23, selectedClfs=None, overwriteSavedResult=True, verbose=True,
                   cvSplitNum=10, test_size=0.25, random_states=[None], **fitArgs):
    """ fits a list of classifiers by searching for the best parameters using GridSearchCV
    @type trainData DatasetPair
    @param selectedClfs: which classifiers to fit. if None, fits all.
    @return: (a dictionary of {classifier name: classifier}, the best classifier)
    """

    res = {}
    bestScore = 0
    bestClf = ()
    if selectedClfs and not isinstance(selectedClfs, Iterable): selectedClfs = [selectedClfs]
    intermediateResdir = os.path.join(rootdir, 'intermediate results')

    # ------ fit using gridsearchcv -----------
    for name, v in classifiersToTry.iteritems():
        if selectedClfs and name not in selectedClfs: continue

        pipe, paramsDict = makePipe([fillertoTry, normalizerToTry, (name, classifiersToTry[name])])

        try:
            newpipe, bestParams, score = fitClfWithGridSearch(name, pipe, paramsDict, trainData, intermediateResdir, useJJ=useJJ,
                                                              n_jobs=n_jobs, overwriteSavedResult=overwriteSavedResult, verbose=verbose,
                                                              cvSplitNum=cvSplitNum, test_size=test_size, random_states=random_states, **fitArgs)

            cleanPipe = pipe.set_params(**bestParams)
            res[name] = cleanPipe

            # check if it's the best classifier
            if score > bestScore:
                bestScore = score
                bestClf = (name, cleanPipe, score)

        except Exception as e:
            print 'Fitting', name, 'caused an error:', e

    return res, bestClf

def buildModel(data, testData, fieldMaps, n_jobs, useJJ, selectedClfs = None, colNames = 'all', random_states=[None],
               writeResults=True, cvNumSplits=50, test_size=0.25, verbose=False, **fitArgs):
    """
    @type data DatasetPair
    @type testData DatasetPair
    @type fieldMaps dict
    @param selectedClfs: the classifiers to run. if None, runs all classifiers
    @param colNames: if 'all' no splicing is done; otherwise is a list of fields to splice by
    @return: test results
    @rtype: Iterable
    """

    pprint({k:fieldMaps[k] for k in colNames})
    print fieldMaps.keys()

    t0 = time()
    res_all = {}
    bestClf_by_split = {}   # {colVals: bestClf}. If bestClf is a scalar, just use it as predictions regardless of the input

    # ------- set up data -------
    if colNames=='all':
        colIndices = range(len(data.fieldNames))
        splits_all = {'all': data}
        splits_test = {'all': testData}
    elif isinstance(colNames, Iterable):
        colIndices = [data.fieldNames.index(name) for name in colNames]
        splits_all = data.spliceByColumnNames(colNames, removeColumns=True)
        splits_test = testData.spliceByColumnNames(colNames, removeColumns=True)
    else:
        raise ValueError("colNames of type %s isn't one of the recognized types." % type(colNames))


    # ------- fit classifiers -------
    for colVals in splits_all.keys():

        colVal_names = 'all' if colNames=='all' else tuple(reverseDict(fieldMaps[name])[colVal] if name in fieldMaps else colVal for name, colVal in zip(colNames, colVals))

        # if not colVal_names==('female', 'Mrs', ''):
        #     continue

        print '='*10, colVal_names, '='*10, splits_all[colVals].dataCount, 'training data.'
        if colVal_names == ('female', 'other', 'Q'):
            print mask2DArrayByCol(testData.X, dict(zip(colIndices, colVals)))[1]
            print splits_all[colVals].X, splits_all[colVals].Y, splits_test[colVals].X, splits_test[colVals].Y


        if splits_all[colVals].dataCount == 0:
            if colVals not in splits_test or splits_test[colVals].dataCount==0:
                print 'Irrelevant category. Skipping...'
            else:   # there is training data but no testing data. use the mode of training data results
                v = type(data.Y[0])(mode(data.Y)[0])
                res_all[colVals] = np.repeat(v, splits_test[colVals].dataCount)
                bestClf_by_split[colVals] = v
                print 'Nothing training data. Using the mode of all training Y values', v
            continue
        elif colVals not in splits_test or splits_test[colVals]==0:
            print 'No testing data for this category. Skipping...'
            continue

        # get this slice's data
        train_cur = splits_all[colVals]
        test_cur = splits_test[colVals]
        print '%s has %d training data, %d testing data' % (colVal_names, train_cur.dataCount, test_cur.dataCount)

        # fit
        if len(np.unique(train_cur.Y))==1 or train_cur.dataCount<=5:  # nothing to fit if the training data has only one class
            v = type(train_cur.Y[0])(mode(train_cur.Y)[0])
            print 'Using the most common one class (%d) in training data for prediction.' % v
            res_all[colVals] = np.repeat(v, test_cur.dataCount)
            bestClf_by_split[colVals] = v

        else:
            _, (bestClfName, bestClf, bestscore) = fitClassifiers(train_cur, selectedClfs=selectedClfs, random_states=random_states,
                                                                  useJJ=useJJ, n_jobs=n_jobs, overwriteSavedResult=True,
                                                                  cvSplitNum=cvNumSplits, test_size=test_size, verbose=verbose, **fitArgs)
            print '>>>>>>> The best classifier for %s is %s, with score %f.' % (colVal_names, bestClfName, bestscore)
            res_all[colVals] = bestClf.fit(*train_cur.getPair()).predict(test_cur.X)
            bestClf_by_split[colVals] = bestClf

    # ------- compute overall cv score ---------
    cvResults = []

    for randomState in random_states:
        cvObj = StratifiedShuffleSplit(data.Y, cvNumSplits, test_size=test_size, random_state=randomState)

        for trainInds, testInds in cvObj:
            if colNames=='all':
                curTrainDataSplitted = {'all': DatasetPair(data.X[trainInds], data.Y[trainInds], data.fieldNames)}
                curTestDataSplitted = {'all': DatasetPair(data.X[testInds], data.Y[testInds], data.fieldNames)}
            else:
                curTrainDataSplitted = DatasetPair(data.X[trainInds], data.Y[trainInds], data.fieldNames).spliceByColumnNames(colNames, removeColumns=True)
                curTestDataSplitted = DatasetPair(data.X[testInds], data.Y[testInds], data.fieldNames).spliceByColumnNames(colNames, removeColumns=True)

            curTotalCount = len(testInds)
            curScore = 0

            for colVals in curTrainDataSplitted.keys():
                if colVals not in bestClf_by_split or curTestDataSplitted[colVals].dataCount==0 or curTrainDataSplitted[colVals].dataCount==0:
                    continue

                trainD = curTrainDataSplitted[colVals]
                testD = curTestDataSplitted[colVals]
                clf = deepcopy(bestClf_by_split[colVals])

                if isinstance(clf, (int, float)) or len(np.unique(trainD.Y))==1:
                    ypred = [clf] * len(testD.Y)
                else:
                    ypred = clf.fit(*trainD.getPair()).predict(testD.X)

                curScore += accuracy_score(testD.Y, ypred) * testD.dataCount / curTotalCount

            cvResults.append(curScore)

    cvScore = np.mean(cvResults)
    print 'OVERALL CV SCORE =', cvScore

    # ------- collect results -------
    if colNames=='all':
        testRes = res_all['all']
    else:
        testRes = np.repeat(99, testData.dataCount)
        for colVals, res in res_all.iteritems():
            _, curMask = mask2DArrayByCol(testData.X, dict(zip(colIndices, colVals)))
            testRes[curMask] = res

    # print testRes
    #
    # print 'jjjjjjjjjjj', list(testRes).index(99), testData.X[list(testRes).index(99)]
    # for i in range(len(testData.fieldNames)):
    #     n = testData.fieldNames[i]
    #
    #     if n in fieldMaps:
    #         print n, ':', fieldMaps[n][testData.X[list(testRes).index(99)][i]]
    #     else:
    #         print n, ':', testData.X[list(testRes).index(99)][i]

    assert np.logical_or(testRes==0, testRes==1).all()    # make sure all values are filled

    # ------- featureSelectionOutput results -------
    if writeResults: writeTestingResToFile("by" + '_'.join(colNames), testRes)

    print 'Total amount of time spent:'
    printDoneTime(t0)

    return testRes, cvScore

def writeTestingResToFile(clfName, testingResults):
    now = datetime.now()
    outputFname = os.path.join(rootdir, 'results',  '%s_%s-%s_%s'%(now.date(), now.hour, now.minute, clfName))
    outputTestingResultsToFile([int(v) for v in testingResults], outputFname)
