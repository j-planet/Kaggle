import sys
from warnings import filterwarnings

from sklearn.pipeline import Pipeline
from sklearn.ensemble.gradient_boosting import LeastSquaresError

from utilities import Normalizer, MissingValueFiller, cvScores
from GradientBoost_JJ import GradientBoost_JJ
from titanicutilities import readData
from globalVariables import svc_f, svc_m, rf_f, rf_m, gb, rootdir

sys.path.append(rootdir)
filterwarnings('ignore')

# def cvScoreInnerLoop(clf):
#     global X, y
#     return cvScores(clf, X, y, n_jobs=23, scoreFuncsToUse='accuracy_score', numCVs=5, doesPrint=False)

def init(*args):
    global X, y
    X, y = args

def checkClfPerformanceByCV():
    data, testData, fieldMaps, sampleWeights, testSampleWeights = readData(outputDir=rootdir)
    random_state = 0

    trans = Pipeline([('filler', MissingValueFiller()), ('normer', Normalizer())])
    X = trans.fit_transform(data.X, data.Y)
    y = data.Y

    verb = 1
    gbjjn_jobs = 10
    maxNumIt = 300
    clfs = [
        GradientBoost_JJ(learners=[svc_f, svc_m, rf_f, rf_m,gb], verbosity=verb, subsample=0.75, learningRate=0.5,
                         numIterations=maxNumIt, lossFunction=LeastSquaresError(1), stoppingError=0.1, numDeemedStuck=15, n_jobs=gbjjn_jobs),
        GradientBoost_JJ(learners=[svc_f, svc_m, rf_f, rf_m,gb], verbosity=verb, subsample=0.5, learningRate=0.5,
                         numIterations=maxNumIt, lossFunction=LeastSquaresError(1), stoppingError=0.1, numDeemedStuck=15, n_jobs=gbjjn_jobs),
        GradientBoost_JJ(learners=[svc_f, svc_m, rf_f, rf_m,gb], verbosity=verb, subsample=0.75, learningRate=0.05,
                         numIterations=maxNumIt, lossFunction=LeastSquaresError(1), stoppingError=0.1, numDeemedStuck=15, n_jobs=gbjjn_jobs),
        GradientBoost_JJ(learners=[svc_f, svc_m, rf_f, rf_m,gb], verbosity=verb, subsample=0.5, learningRate=0.05,
                         numIterations=maxNumIt, lossFunction=LeastSquaresError(1), stoppingError=0.1, numDeemedStuck=15, n_jobs=gbjjn_jobs),
        GradientBoost_JJ(learners=[svc_f, svc_m, rf_f, rf_m,gb], verbosity=verb, subsample=0.5, learningRate=0.01,
                         numIterations=maxNumIt, lossFunction=LeastSquaresError(1), stoppingError=0.1, numDeemedStuck=15, n_jobs=gbjjn_jobs),
        GradientBoost_JJ(learners=[svc_f, svc_m, rf_f, rf_m,gb], verbosity=verb, subsample=0.5, learningRate=0.01,
                         numIterations=maxNumIt, lossFunction=LeastSquaresError(1), stoppingError=0.07, numDeemedStuck=15, n_jobs=gbjjn_jobs)]

    clfs = clfs[2:]
    res = [cvScores(clf, X, y, n_jobs=24, scoreFuncsToUse='accuracy_score', numCVs=10, doesPrint=False) for clf in clfs]

    # pool = MyPool(processes=10, initializer=init, initargs=(X, y))
    # temp = pool.map_async(cvScoreInnerLoop, clfs)
    # temp.wait()
    # res = temp.get()

    print res
    for i in range(len(clfs)):
        print '=' * 10
        print clfs[i]
        print res[i]

if __name__ == "__main__":
    checkClfPerformanceByCV()
    data, testData, fieldMaps, sampleWeights, testSampleWeights = readData(outputDir=rootdir)
    # random_state = 0
    #
    # buildModel(data, testData, fieldMaps, selectedClfs=['JJGradientBoostsimple'], n_jobs=4, writeResults=True,
    #            colNames='all', cvNumSplits=10, random_state=random_state)

    print '\n>>>>>>>>>>> Done <<<<<<<<<<<<<<'