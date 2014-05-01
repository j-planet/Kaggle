import sys
from pprint import pprint
sys.path.append('/home/jj/code/Kaggle/allstate')

from sklearn.preprocessing import Imputer, Normalizer
from sklearn.metrics import accuracy_score

from Kaggle.utilities import jjcross_val_score, DatasetPair
from Kaggle.CV_Utilities import fitClfWithGridSearch
from helpers import *
from pipes import make_pipes
from RiskFactors import ImputerJJ


calibrationName = 'smallerTrain'
trainingName = 'train'

# ======= CALIBRATE CLASSIFIERS =======
_, inputTable_cal, outputTable_cal, _ = condense_data(calibrationName, isTraining=True, readFromFiles=True)
pdf(inputTable_cal)

riskFactorImp = ImputerJJ(inputTable_cal)
temp = riskFactorImp.fit_transform(inputTable_cal)
X_cal = Normalizer().fit_transform(temp)
# X_cal = Normalizer().fit_transform(Imputer().fit_transform(inputTable_cal))
y_cal = CombinedClassifier.combine_outputs(np.array(outputTable_cal))


# plot_feature_importances(X_train, outputTable, inputTable.columns)

print '----------- individual accuracy score'

indivClfs = []

for col in outputTable_cal.columns:
    print '>'*20, col, '<'*20
    cur_y = np.array(outputTable_cal[col])

    bestScore = -1
    bestPipe = None
    bestParams = None

    for name, (pipe, params) in make_pipes().iteritems():
        print '>'*10, name, '<'*10

        _, cur_bestParams, cur_bestScore = fitClfWithGridSearch(
            '_'.join([name, col, calibrationName]), pipe, params, DatasetPair(X_cal, cur_y),
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

    indivClfs.append(bestPipe)
    print '---->', col, '<----', bestScore
    pprint(bestParams)

combinedClf = CombinedClassifier(indivClfs)
print 'OVERALL CV SCORE:', np.mean(jjcross_val_score(combinedClf, X_cal, y_cal, accuracy_score, cv=5, n_jobs=N_JOBS)) # validate classifier

print '====== TRAINING'

_, inputTable_train, outputTable_train, _ = condense_data(trainingName, isTraining=True, readFromFiles = True)
pdf(inputTable_train)

X_train = Normalizer().fit_transform(riskFactorImp.fit_transform(inputTable_train))
# X_train = Normalizer().fit_transform(Imputer().fit_transform(inputTable_train))
y_train = CombinedClassifier.combine_outputs(np.array(outputTable_train))

combinedClf.fit(X_train, y_train)



print '====== PREDICTING'
isValidation = False
_, inputTable_test, _, combinedTable_test = condense_data('test_v2', isTraining=isValidation, readFromFiles=True)
pdf(inputTable_test)

# X_test = Normalizer().fit_transform(Imputer().fit_transform(inputTable_test))
X_test = Normalizer().fit_transform(riskFactorImp.fit_transform(inputTable_test))
preds = combinedClf.predict(X_test)

if isValidation:
    combinedTable_test['PRED'] = preds
    combinedTable_test.to_csv(DATA_DIR + 'val.csv')

# --- write out to file
res = pandas.DataFrame(index = inputTable_test.index, data = preds)
res.columns = ['plan']
res.to_csv('/home/jj/code/Kaggle/allstate/submissions/test.csv')