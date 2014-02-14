"""
GridSearch CV using Genetic Algorithm
"""

__author__ = 'jjin'

from utilities import makePipe, jjcross_val_score, MyPool, printDoneTime, getNumCvFolds, saveObject, runPool
# from titanic.titanicutilities import readData
from titanic.globalVariables import rootdir, fillertoTry, normalizerToTry, classifiersToTry

import numpy as np
from copy import copy
from time import time
from random import choice, sample, uniform
from pprint import pprint
from itertools import product

from scipy import isinf, isnan, isscalar
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn import clone
from pybrain.optimization import GA
from pybrain.utilities import DivergenceError

def GAGridSearchCV_JJ_learnStepInit(*args):
    """
    called in _learnStep of GAGridSearchCV_JJ
    @param args: evaluator
    """

    global allCvData, origpipe, allParamsDict, scoreFunc
    allCvData, origpipe, allParamsDict, scoreFunc = args

def GAGridSearchCV_JJ_learnStepInner(args):
    """ called in _learnStep of GAGridSearchCV_JJ. evaluates one evaluator.
    @return fitness
    """

    global allCvData, origpipe, allParamsDict, scoreFunc
    evaluable, cvIndex = args

    trainX, trainY, testX, testY = allCvData[cvIndex]

    params = getParamsFromIndices(evaluable, allParamsDict)
    newpipe = clone(origpipe)
    newpipe.set_params(**params)
    newpipe.fit(trainX, trainY)

    # res = accuracy_score(testY, newpipe.predict(testX))
    res = scoreFunc(testY, newpipe.predict(testX))

    return res


class GAGridSearchCV_JJ(GA):
    """
    adapts GA to gridsearchcv
    Instead of strings, each individual is a list of non-negative integers representing the index
    """

    def __init__(self, data, pipe, allParamsDict, cvs,
                 minimize, maxLearningSteps, populationSize,
                 eliteProportion, parentsProportion, maxValsForInputs, initialPopulation, mutationProportion, mutationProbability,
                 convergenceTolerance = 0, mutationStdDev=None, initialEvaluables=None, numConvergenceSteps=None, n_jobs=1,
                 verbosity=1, saveCache=True, maxDuplicateProportion=0, scoreFunc=accuracy_score, numCvFolds=None, **kargs):
        """
        @param data
        @param minimize whether to min or max the function
        @type minimize bool
        @param populationSize (constant) population size
        @param eliteProportion: portion of elites. a decimal in [0, 1]
        @param parentsProportion: portion of the population to use as parents at each stage. a decimal in [0,1]
        @param mutationProbability a decimal in [0,1]. mutation probability
        @param mutationStdDev: how much to deviate from the original when mutating
        @param initialEvaluables a list of inputs
        @param numConvergenceSteps if results have been increasing/decreasing for this many steps, then we say it has converged
        @param maxValsForInputs a list of values presenting the max for each index of an evaluable
        @param n_jobs: number of processes used in evaluating individuals at each learning step
        @param cv_n_jobs: the n_jobs parameter for jj_cv_score
        @param saveCache: whether to save previous results
        @param maxDuplicateProportion: highest allowable portion of duplicate individuals
        @param cvs: a list of cv objects
        @param scoreFunc: the score function to optimize
        """

        assert numConvergenceSteps is None or numConvergenceSteps >= 2

        # members that are not in the base class
        self.bestOutcomes = []   # the best (fitness, individual) for each stage
        self._pool = None        # a list of pools, one for each individual
        self._allCvData = None
        self._resultsCache = {}   # a dictionary of {indiv:evaluation} for caching/performance purpose

        self._data = data
        self._pipe = pipe
        self._allParamsDict = allParamsDict
        self._cvs = cvs
        self._numCvFolds = getNumCvFolds(self._cvs[0]) if numCvFolds is None else numCvFolds
        self.maxLearningSteps = maxLearningSteps
        self.populationSize = populationSize
        self.elitism = True
        self.eliteProportion = eliteProportion
        self.topProportion = parentsProportion
        self.mutationProbability = mutationProbability
        self.mutationStdDev = mutationStdDev
        self.mutationProportion = mutationProportion
        self._numConvergenceSteps = numConvergenceSteps
        self._convergenceTolerance = convergenceTolerance
        self._maxValsForInputs = maxValsForInputs
        self._verbosity = verbosity
        self.initialPopulation = initialPopulation
        GA.__init__(self, self._oneEvaluation, initialEvaluables, **kargs)
        self.minimize = minimize
        self.bestParams = None
        self._saveCache = saveCache
        self._maxDuplicateCount = int(round(populationSize * maxDuplicateProportion))
        self.score_func = scoreFunc

        # dunno what these do. just set to false for now...
        self._wasUnwrapped = False
        self._wasWrapped = False

        # make pool for multiprocessing

        if n_jobs > 1:
            self._n_jobs = max(1, min(n_jobs, self._numCvFolds * len(self._cvs) * self.populationSize))    # the number of processes for each individual's cv calculation

            t0 = time()

            self._allCvData = []  # list of (trainx, trainy, testx, testy)

            for cv in self._cvs:
                for trainInds, testInds in cv:    # all cv data
                    trainX = self._data.X[trainInds]
                    trainY = self._data.Y[trainInds]
                    testX = self._data.X[testInds]
                    testY = self._data.Y[testInds]
                    self._allCvData.append((trainX, trainY, testX, testY))

            self._pool = MyPool(processes = self._n_jobs, initializer=GAGridSearchCV_JJ_learnStepInit,
                                 initargs=(self._allCvData, self._pipe, self._allParamsDict, self.score_func))
            printDoneTime(t0, 'Making the pool')

        elif n_jobs==1:
            self._n_jobs = 1
        else:
            raise Exception('Invalid number of jobs, %d' % n_jobs)

    def _oneEvaluation(self, indiv):    # only called in single process mode

        if self._saveCache and tuple(indiv) in self._resultsCache: # has been calculated before => read from cache
            print indiv, 'has been calculated before. Reading evaluation from cache.'
            res = self._resultsCache[tuple(indiv)]

        else:
            params = getParamsFromIndices(indiv, self._allParamsDict)
            newpipe = clone(self._pipe)
            newpipe.set_params(**params)

            res = 0
            for cv in self._cvs:
                # res += jjcross_val_score(newpipe, self._data.X, self._data.Y, accuracy_score, cv=cv, n_jobs=1).sum()
                res += jjcross_val_score(newpipe, self._data.X, self._data.Y, self.score_func, cv=cv, n_jobs=1).sum()

            res /= sum(getNumCvFolds(cv) for cv in self._cvs)

            if isscalar(res) and (isnan(res) or isinf(res)): raise DivergenceError # detect numerical instability

        if self._verbosity >= 2:
            print 'Evaluation for', indiv, 'is', res

        return res

    def initPopulation(self):

        if self.initialPopulation is None:
            # complete random guess for the initial population
            self.currentpop = [self._initEvaluable] + \
                [mutateIndiv(self._initEvaluable, self._maxValsForInputs, self.mutationProportion, self.mutationStdDev) for _ in range(self.populationSize-1)]

        else:
            self.currentpop = self.initialPopulation

    def mutated(self, indiv):
        """ mutate some genes of the given individual """
        if np.random.random() < self.mutationProbability:
            return mutateIndiv(indiv, self._maxValsForInputs, self.mutationProportion, self.mutationStdDev)

        else:
            return copy(indiv)

    @property
    def selectionSize(self):    # used as parents for reproduction. at least 2
        return max(int(round(self.populationSize * self.topProportion)), 2)

    @property
    def eliteSize(self):
        """ the number of elites. at least one. determined by population size * elite proportion
        """

        if self.elitism:
            return max(int(round(self.populationSize * self.eliteProportion)), 1)  # at least one elite
        else:
            return 0

    def crossOver(self, parents, nbChildren):
        """ generate a number of children by doing 1-point cross-over """

        xdim = self.numParameters

        if xdim<2:
            return [choice(parents) for _ in range(nbChildren)]

        else:
            children = []

            for _ in range(nbChildren):
                p1, p2 = sample(parents, 2)                     # pick 2 parents
                point = choice(range(xdim-1))                   # where to cross over
                newChild = np.zeros(xdim, dtype=int)
                newChild[:point] = p1[:point]
                newChild[point:] = p2[point:]
                children.append(newChild)

            return children

    def produceOffspring(self):
        """ produce offspring by selection, mutation and crossover. """

        numElites = min(self.eliteSize, self.selectionSize)

        parents = self.select()
        elites = parents[:numElites]

        if self._verbosity >= 2:
            print '--- parents:'
            pprint(parents)
            print '--- elites:'
            pprint(elites)

        self.currentpop = elites + [self.mutated(child) for child in self.crossOver(parents, self.populationSize-numElites)]

        # --- make sure there are no twins in the population ---
        toContinue = True

        while toContinue:
            duplicateIndices = []

            # find duplicates
            for i in range(len(self.currentpop)):
                for j in np.arange(i+1, len(self.currentpop)):
                    if np.array_equal(self.currentpop[i], self.currentpop[j]):
                        if self._verbosity >= 3: print str(i) + ' and ' + str(j) + ' are twins!'
                        duplicateIndices.append(j)

            # re-make duplicates
            if len(duplicateIndices) > self._maxDuplicateCount:
                if self._verbosity >= 3: print '--> adding %d new children '%len(duplicateIndices)
                newChildren = self.crossOver(parents, len(duplicateIndices))
                for i in range(len(newChildren)):
                    self.currentpop[duplicateIndices[i]] = self.mutated(newChildren[i])
            else:
                toContinue = False

    def _saveTheBest(self):
        """ store the best in self.bestEvaluation, self.bestEvaluable and self.bestOutcomes
        """

        self.bestEvaluation = min(self.fitnesses) if self.minimize else max(self.fitnesses)
        bestInd = self.fitnesses.index(self.bestEvaluation)

        self.bestEvaluable = self.currentpop[bestInd].copy()
        self.bestParams = getParamsFromIndices(self.bestEvaluable, self._allParamsDict)
        self.bestOutcomes.append((self.bestEvaluation, self.bestEvaluable))

    def _saveAll(self):
        """if desired, also keep track of all evaluables and/or their fitness."""

        if self._saveCache:
            # print 'x'*10
            # print self.currentpop
            # print self.fitnesses
            # print zip(self.currentpop, self.fitnesses)
            self._resultsCache.update(dict(zip([tuple(p) for p in self.currentpop], self.fitnesses)))   # cache results

        if self.storeAllEvaluated:
            self._allEvaluated += [ev.copy() for ev in self.currentpop]

        if self.storeAllEvaluations:
            if self._wasOpposed and isscalar(self.fitnesses[0]):
                self._allEvaluations += [-r for r in self.fitnesses]
            else:
                self._allEvaluations += self.fitnesses

        if self.storeAllPopulations:
            self._allGenerations.append((self.currentpop, self.fitnesses))

    def _learnStep(self):
        """ do one generation step """
        if self._n_jobs == 1:
            if self._verbosity >=1 : print 'GA_JJ in Single thread'

            self.fitnesses = []
            self.fitnesses = [self._oneEvaluation(indiv) for indiv in self.currentpop]
        else:
            if self._verbosity >=1:print 'GA_JJ in multi thread with %d jobs' % self._n_jobs

            if self._resultsCache:  # read from cache
                # calculate for individuals that haven't been evaluated before
                indexOfPopToCalc = [i for i in xrange(self.populationSize) if tuple(self.currentpop[i]) not in self._resultsCache]
                popToCalc = [self.currentpop[i] for i in indexOfPopToCalc]
                inputData = [v for v in product(popToCalc, range(self._numCvFolds))]  # population x numCvFolds
                res = runPool(self._pool, GAGridSearchCV_JJ_learnStepInner, inputData)    # temporary results: [p1_fold1, ..., p1_fold5, ..., p10_fold1,... p10_fold5]
                popCalcRes = [res[i*self._numCvFolds : (i*self._numCvFolds + self._numCvFolds)].mean() for i in xrange(len(popToCalc))]

                # collect results
                self.fitnesses = []
                for i in xrange(self.populationSize):
                    if i in indexOfPopToCalc:
                        self.fitnesses.append(popCalcRes[indexOfPopToCalc.index(i)])
                    else:
                        self.fitnesses.append(self._resultsCache[tuple(self.currentpop[i])])

                if self._verbosity == 2:
                    print '--- current population:'
                    pprint(self.currentpop)
                    print '--- population to be calculated:'
                    pprint(popToCalc)
                    print '--- population read from cache:'
                    pprint([self.currentpop[i] for i in set(range(self.populationSize)) - set(indexOfPopToCalc)])

            else:   # all calculate from scrach
                inputData = [v for v in product(self.currentpop, range(self._numCvFolds))]  # population x numCvFolds
                res = runPool(self._pool, GAGridSearchCV_JJ_learnStepInner, inputData)    # temporary results: [p1_fold1, ..., p1_fold5, ..., p10_fold1,... p10_fold5]
                self.fitnesses = [res[i*self._numCvFolds : (i*self._numCvFolds + self._numCvFolds)].mean() for i in xrange(len(self.currentpop))]

            if self._verbosity >= 3:    # for debugging purposes
                print 'x'*20, 'in _learnStep'
                print 'inputData:'
                pprint(inputData)
                pprint(zip(inputData, res))

        if self._verbosity == 2:
            print '>>>>> Results for this round:'
            pprint(zip(self.currentpop, self.fitnesses))
        elif self._verbosity >= 3:
            print '>>>>> Results for this round:'
            pprint(zip(self.currentpop, [getParamsFromIndices(p, self._allParamsDict) for p in self.currentpop], self.fitnesses))


        self.numEvaluations += len(self.currentpop)
        self._saveTheBest()
        self._saveAll()

        # double mutation when results stag
        if len(self.bestOutcomes)>1 and self.bestOutcomes[-1][0] <= self.bestOutcomes[-2][0]:

            self.mutationProbability = min(1.0, 1.2*self.mutationProbability)
            self.mutationProportion = min(1.0, 1.2*self.mutationProportion)

            if self.mutationStdDev is not None and self.mutationStdDev>0:
                self.mutationStdDev *= 1.2
            elif self.mutationStdDev==0:
                self.mutationStdDev = None

            self._maxDuplicateCount /= 2.0

            if self._verbosity >= 1:
                print '>>> Amping up mutation!!'
                print 'Now parents proportion = %0.3f, mutation probability = %0.3f, mutation proportion = %0.3f, max dup count = %d' % (self.topProportion, self.mutationProbability, self.mutationProportion, self._maxDuplicateCount)


        self.produceOffspring()

    def _stoppingCriterion(self):
        # base class check
        if GA._stoppingCriterion(self):
            print '====== STOPPING DUE TO number of steps ====='
            return True

        # check for convergence
        l = [v[0] for v in self.bestOutcomes]
        n = len(l)
        if self._numConvergenceSteps>n: return False

        for i in np.arange(n - self._numConvergenceSteps, n-1):
            if self._wasOpposed and l[i]-l[i+1]>self._convergenceTolerance:
                return False

            if not self._wasOpposed and l[i+1]-l[i]>self._convergenceTolerance:
                return False

        print '====== stopping due to convergence (%d steps, %0.3f tolerance) =====' % (self._numConvergenceSteps, self._convergenceTolerance)
        return True

    def _notify(self):
        """ Provide some feedback during the run. """
        if self._verbosity >=1:
            print '======= Step:', self.numLearningSteps, 'best evaluation:', self.bestEvaluation, '; best individual:', self.bestEvaluable
        if self.listener is not None:
            self.listener(self.bestEvaluable, self.bestEvaluation)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pool:
            if self._verbosity >=1: print 'Draining the pool. no more swimming :('

            self._pool.close()
            self._pool.join()


def mutateIndiv(indiv, maxValsForInputs, mutationProportion, mutationStdDev):
    """
    mutates an input to within a range
    @param indiv: the individual (numpy array) to be mutated
    @param maxValsForInputs: max values for each position of an individual
    @param mutationProportion: how many (in decimals) of positions to mutate
    @param mutationStdDev: how much (in decimal) to mutate by. x -> x * (1 +/- stdev)
    @return: a new np.array object
    """
    n = len(indiv)
    assert n==len(maxValsForInputs)

    res = np.empty(n, dtype=int)
    posToChange = sample(range(n), int(round(n * mutationProportion)))

    for i in range(n):
        if i in posToChange:
            if mutationStdDev is None:
                l = range(maxValsForInputs[i] + 1)
                l.remove(indiv[i])      # cannot "mutate" to its original value
                res[i] = choice(l)
            else:
                res[i] = min(max(int(round(uniform(1-mutationStdDev, 1+mutationStdDev) * indiv[i])), 0), maxValsForInputs[i])
        else:
            res[i] = indiv[i]

    print 'mutating:', indiv, '--->', res
    return res

def generateInputs(maxValsForInputs, count=1):
    """
    Randomly generate an input for GAGridSearchCV_JJ
    @param maxValsForInputs: a list of values presenting the max for each index of an evaluable
    @param count: number of inputs to be generated
    @return: a list of inputs
    """

    if count==1:
        return np.array([np.random.random_integers(low=0, high=v) for v in maxValsForInputs])
    else:
        allPos = [v for v in product(*[range(i+1) for i in maxValsForInputs])]
        return np.array([np.array(v) for v in sample(allPos, count)])

def getParamsFromIndices(indices, allParamsDict):
    return dict((k, v[indices[i]]) for i,(k,v) in enumerate(allParamsDict.iteritems()))


def fakeOneEvaluation(indiv, allParamsDict, pipe, data, cv):    # only called in single process mode

    params = getParamsFromIndices(indiv, allParamsDict)
    newpipe = clone(pipe)
    newpipe.set_params(**params)

    res = jjcross_val_score(newpipe, data.X, data.Y, accuracy_score, cv=cv, n_jobs=1).mean()
    if isscalar(res) and (isnan(res) or isinf(res)): raise DivergenceError # detect numerical instability

    print 'Evaluation for', indiv, 'is', res

    return res

# if __name__ == '__main__':
#     data, _, _, _, _ = readData(outputDir=rootdir)
#     #
#     # # make pipe and allParamsDict
#     name = 'svc'
#     step1 = fillertoTry
#     step2 = normalizerToTry
#     step3 = (name,classifiersToTry[name])
#     pipe, allParamsDict = makePipe([step1, step2, step3])
#     pprint(allParamsDict)
#
#     maxValues = [len(x)-1 for x in allParamsDict.values()]
#     print maxValues
#     maxPopSize = np.prod([v+1 for v in maxValues])
#
#     for _ in range(10):
#         mutateIndiv(indiv= np.array([2, 0, 0, 0, 0, 2, 0, 0]),maxValsForInputs=np.array([3, 1, 3, 2, 1, 2, 1, 2]), mutationProportion=1, mutationStdDev=0.3)
#
#      # ---> [2 0 3 0 0 1 1 0]
#
#     print 'maxPopSize =', maxPopSize
#     randomState = 0
#     numCvFolds = 3
#     cvObj = StratifiedShuffleSplit(data.Y, numCvFolds, test_size=0.25, random_state=randomState)
#     # popSize = min(10, maxPopSize)
#     # initPop = generateInputs(maxValues, count=popSize)
#     popSize = 2
#     initPop = np.array([np.array([0, 0, 3, 0, 0, 0, 0, 1]), np.array([0,0,0,1,0,0,0,1])])
#
#     # fakeOneEvaluation(np.array([1,0,0]), allParamsDict, pipe, data, cvObj)
#
#     t0 = time()
#     with GAGridSearchCV_JJ(data=data, pipe=pipe, allParamsDict=allParamsDict, cv=cvObj, minimize=False, maxValsForInputs=maxValues,
#                             initialEvaluables=initPop[0], initialPopulation=initPop, populationSize=popSize, verbosity=2,
#                             maxLearningSteps=10, numConvergenceSteps=5, convergenceTolerance=0, eliteProportion=0.1,
#                             parentsProportion=0.4, mutationProbability=0, mutationStdDev=None, n_jobs=10) \
#     as ga:
#         ga.learn()
#
#         bestParams = getParamsFromIndices(ga.bestEvaluable, allParamsDict)
#
#         print ga.bestEvaluable
#         print bestParams
#         print ga.bestEvaluation
#
#         newpipe = clone(pipe)
#         newpipe.set_params(**bestParams)
#         print jjcross_val_score(newpipe, data.X, data.Y, accuracy_score, cv=cvObj, n_jobs=1).mean()
#
#     printDoneTime(t0, '---- GA')