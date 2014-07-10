__author__ = 'jjin'

""" A gradient boost class that can take a list of distinct learners.
"""

from warnings import warn
from time import time
import random

from sklearn import clone
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble.gradient_boosting import *
from sklearn.cross_validation import KFold
from scipy.optimize import minimize_scalar

from utilities import jjcross_val_score, splitTrainTest, MajorityPredictor, printDoneTime
from pool_JJ import MyPool


def gbjjInnerLoop(learner):
    """
    @param learner:
    @return: cv score of the learner
    """
    global trainX, trainY, lf, n_jobs, cvObj

    l = clone(learner)
    scores = jjcross_val_score(l, trainX, trainY, score_func=lf, n_jobs=n_jobs, cv=cvObj)
    return scores.mean()


def gbjjInit(*args):
    global trainX, trainY, lf, n_jobs, cvObj
    trainX, trainY, lf, n_jobs, cvObj = args


class GradientBoost_JJ(BaseEstimator, TransformerMixin):

    def __init__(self, learners, numIterations, lossFunction, subsample=1.0, verbosity=1, learningRate=0.5,
                 stoppingError=0, numDeemedStuck=20, n_jobs=1, randomState=None, cvNumFolds=5):
        """
        @type lossFunction LossFunction
        @type learners list
        @type learningRate float
        @type stoppingError float
        @param subsample: portion of the MLWave to fit with at each stage. If <1.0 then stochastic gradient boosting. prevent overfitting.
        @param learners: a list of learners to be considered at each stage
        @param numIterations: if None iterate until stoppingError has been met
        @param lossFunction:
        @param learningRate: regularization factor, the learning rate. in (0, 1]
        @param stoppingError: the error below (<=) which to stop. 0 by default (i.e. never stop early)
        @param numDeemedStuck: if the number of rounds during which error stays the same, then quit fitting
        @param randomState: used in cvObj (for jjcvscore) and as the seed for subsampling
        @return:
        """

        assert 0< learningRate <=1.0, 'The learning rate (%f) is not in (0, 1].' % learningRate
        assert isinstance(lossFunction, LossFunction)
        if numIterations is None and stoppingError==0:
            warn("Neither of number of iterations and stopping error is bounded. This may give rise to indefinite iteration.")

        self.verbosity = verbosity
        self.numIterations = numIterations
        self.lossFunction = lossFunction
        self._estimators = []   # [(estimator, weight), ...]
        self.learners = learners
        self.learningRate = learningRate
        self.stoppingError = stoppingError
        self.numDeemedStuck = numDeemedStuck
        self._currentPrediction = None
        self.subsample = subsample
        self.n_jobs = min(n_jobs, len(self.learners))
        self.randomState = randomState
        self.cvNumFolds = cvNumFolds

        # own random with pre-determined seed
        self.rnd = random.Random()
        self.rnd.seed(self.randomState)


    def _fit_stage(self, X, y, rmTolerance):
        """
        fits one stage of gradient boosting
        @param X:
        @param y:
        @param rmTolerance: tolerance for 1D optimization
        @return: nothing
        """

        residuals = self.lossFunction.negative_gradient(y, self._currentPrediction)
        trainX, trainY, _, _ = splitTrainTest(X, residuals, 1-self.subsample)   # stochastic boosting. train only on a portion of the data

        if len(np.unique(trainY))==1:
            hm = MajorityPredictor().fit(trainY)
        else:
            cvObj = KFold(n=len(trainX), n_folds=self.cvNumFolds, indices=False, shuffle=True, random_state=self.randomState)

            # find the h that best mimics the negative gradient
            if self.n_jobs > 1:  # parallel
                n_jobs = max(1, self.n_jobs/len(self.learners), self.cvNumFolds)
                # n_jobs = 1
                pool = MyPool(processes=self.n_jobs, initializer=gbjjInit, initargs=(trainX, trainY, self.lossFunction, n_jobs, cvObj))
                temp = pool.map_async(gbjjInnerLoop, self.learners)
                temp.wait()
                h_res = temp.get()
                pool.close()
                pool.join()

            else:   # single thread
                h_res = []

                for learner in self.learners:
                    if self.verbosity >= 2:
                        print 'Fitting learner:', learner
                    l = clone(learner)
                    scores = jjcross_val_score(l, trainX, trainY, score_func=self.lossFunction, n_jobs=1, cv=cvObj)
                    h_res.append(scores.mean())

            hm = clone(self.learners[np.argsort(h_res)[0]])

        if self.verbosity>=1:
            print "The best classifier is", hm.__class__

        # find rm
        hm.fit(trainX, trainY)
        hmx = hm.predict(X)
        rm = minimize_scalar(lambda r: self.lossFunction(y, self._currentPrediction + r*hmx), tol=rmTolerance).x

        # append estimator and weight
        self._estimators.append((hm, rm))

    def fit(self, X, y, rmTolerance=1e-8):
        if self.verbosity >= 1:
            if self.n_jobs > 1: print 'GB_JJ in parallel with %d processes' % self.n_jobs
            else: print 'GB_JJ in single thread'
        self._currentPrediction = None
        self._estimators = []

        # --- fit initial estimator ---
        temp = self.lossFunction.init_estimator()
        temp.fit(X,y)
        self._estimators.append((temp, 1))
        self._currentPrediction = self._evaluate(X)
        curError = self.lossFunction(y, self._currentPrediction)

        # --- fit subsequent estimators ---
        numIt = 0
        lastContError = curError
        numContError = 1

        while curError > self.stoppingError and numContError<self.numDeemedStuck and (self.numIterations is None or numIt < self.numIterations):
            if self.verbosity>=1:
                print '---- Fitting stage %d ----' % (numIt + 1)
                print 'Previous error: %f' % curError

            t0 = time()
            self._fit_stage(X, y, rmTolerance)
            printDoneTime(t0)
            self._currentPrediction = self._evaluate(X)

            curError = self.lossFunction(y, self._currentPrediction)

            if curError==lastContError:
                numContError += 1
            else:
                lastContError = curError
                numContError = 1

            numIt += 1

        s = 'iterations = %d, learning rate = %0.2f, subsample = %0.2f, target error = %0.2f' % (numIt+1, self.learningRate, self.subsample, self.stoppingError)

        if numContError==self.numDeemedStuck:
            print s + ' :stopping because error has stayed constant (%f)' % lastContError
        elif self.numIterations is None or numIt < self.numIterations:
            print s + ' :stopping because training error (%f) has dipped below the threshold (%f)' % (curError, self.stoppingError)
        elif numIt>=self.numIterations:
            print s + ' :stopping because the maximum number of iterations (%d) has been reached. Stopped at error = %0.2f' % (self.numIterations, curError)

    def _evaluate(self, X):
        """
        @type X np.array
        @param X:
        @return: a vector of predictions
        @rtype list
        """

        res = self._estimators[0][0].predict(X).ravel() # the "constant" term, F0

        for estimator, weight in self._estimators[1:]:
            res +=  self.learningRate * weight * estimator.predict(X).ravel()

        return res
        # return np.array([round(v) for v in res])    # make the featureSelectionOutput either 0 or 1

    def predict(self, X):
        if len(self._estimators)>0:
            res = self._evaluate(X)
            return np.array([round(v) for v in res])  # make the featureSelectionOutput either 0 or 1

        raise Exception("Please fit the classifier first.")

    def getEstimators(self):
        return self._estimators
