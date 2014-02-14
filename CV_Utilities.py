from utilities import *
from Kaggle import GA_JJ


def fitClfWithGridSearch(name, origpipe, paramDict, data, saveToDir, useJJ,
                         score_func = accuracy_score, test_size=0.25, minimize=False, n_jobs=1, cvSplitNum=10,
                         random_states = [None], cvObjs=None, overwriteSavedResult=True, verbosity=1, **fitArgs):
    """
    tries given parameters for a classifer using GridSearchCV. returns and saves the result to file.
    does NOT change the original pipe input
    @type data DatasetPair
    @param name: name of the classifier
    @param origpipe: the pipe
    @param paramDict: parameters to try
    @param saveToDir: where the intermediate result is saved
    @param test_size: portion of test data in CV (default 0.25). Ignored if cvObjs is not None.
    @param n_jobs: default 1
    @param overwriteSavedResult: if false, always reads from (if) existing file; if true, overwrite only when the score is better
    @return: (pipe, best_params, score)
    """

    print '-'*5, 'Fitting', name, '-'*5
    outputFname = os.path.join(saveToDir, name + '.pk')
    pipe = deepcopy(origpipe)


    if cvObjs is None:
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

            with GA_JJ.GAGridSearchCV_JJ(data=data, pipe=pipe, allParamsDict=paramDict, cvs=cvObjs, minimize=minimize,
                                         maxValsForInputs=maxValues, initialEvaluables=initPop[0],
                                         initialPopulation=initPop, n_jobs=n_jobs, scoreFunc = score_func,
                                         verbosity=verbosity, **fitArgs) \
                as ga:
                ga.learn()

                score = ga.bestEvaluation
                best_params = ga.bestParams
                best_estimator = clone(pipe)
                best_estimator.set_params(**best_params)

                if verbosity >= 1: print_GSCV_info(ga, isGAJJ=True, bestParams=best_params)

        else:           # use GridSearchCV
            score = 0
            for cvObj in cvObjs:
                pipe = GridSearchCV(pipe, paramDict, score_func=accuracy_score, n_jobs=n_jobs, cv=cvObj)

                pipe.fit(*data.getPair())
                best_params = pipe.best_params_
                best_estimator = pipe.best_estimator_
                if verbosity >= 2: print_GSCV_info(pipe, isGAJJ=False, bestParams=best_params)

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
