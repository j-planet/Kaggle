import sys
from pprint import pprint
sys.path.append('/home/jj/code/Kaggle/allstate')

from sklearn.preprocessing import Imputer, Normalizer, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

from Kaggle.utilities import print_missing_values_info, jjcross_val_score, makePipe, DatasetPair
from Kaggle.CV_Utilities import fitClfWithGridSearch
from globalVars import *
from helpers import *


# ======= read data =======
_, inputTable_cal, outputTable_cal, _ = condense_data('train', DATA_DIR, isTraining=True, readFromFiles=True,
                                                                    outputDir= CONDENSED_TABLES_DIR)
X_cal = Normalizer().fit_transform(Imputer().fit_transform(inputTable_cal))  # TODO: better imputation
y_cal = CombinedClassifier.combine_outputs(np.array(outputTable_cal))

pdf(inputTable_cal)
# plot_feature_importances(X_train, outputTable, inputTable.columns)

# ======= CALIBRATE CLASSIFIERS =======

print '----------- individual accuracy score'

indivClfs = []

for col in outputTable_cal.columns:
    cur_y = np.array(outputTable_cal[col])
    clf = GradientBoostingClassifier(subsample=0.8, n_estimators=50, learning_rate=0.05)
    #clf = RandomForestClassifier(n_estimators=25, n_jobs=20)

    pipe, params = makePipe([('GBC', (GradientBoostingClassifier(),
                                      {'learning_rate': [0.01, 0.1, 0.5, 1],
                                       'n_estimators': [5, 10, 25, 50, 100],
                                       'subsample': [0.7, 0.85, 1]}))])

    _, bestParams, score = fitClfWithGridSearch('GBC_' + col, pipe, params, DatasetPair(X_cal, cur_y),
                                                saveToDir='/home/jj/code/Kaggle/allstate/output/gridSearchOutput',
                                                useJJ=True, score_func=accuracy_score, n_jobs=20, verbosity=3,
                                                minimize=False, cvSplitNum=5,
                                                maxLearningSteps=10,
                                                numConvergenceSteps=5, convergenceTolerance=0, eliteProportion=0.1,
                                                parentsProportion=0.4, mutationProportion=0.1, mutationProbability=0.1,
                                                mutationStdDev=None, populationSize=6)

    bestPipe = clone(pipe)
    bestPipe.set_params(**bestParams)

    indivClfs.append(bestPipe)
    print '---->', col, '<----', score
    pprint(bestParams)


print '====== TRAINING'

_, inputTable_train, outputTable_train, _ = condense_data('train', DATA_DIR, isTraining=True, readFromFiles = True,
                                                      outputDir= CONDENSED_TABLES_DIR)
pdf(inputTable_train)
X_train = Normalizer().fit_transform(Imputer().fit_transform(inputTable_train))  # TODO: better imputation
y_train = CombinedClassifier.combine_outputs(np.array(outputTable_train))

combinedClf = CombinedClassifier(indivClfs)
combinedClf.fit(X_train, y_train)

print jjcross_val_score(combinedClf, X_train, y_train, accuracy_score, cv=5, n_jobs=20) # validate classifier


print '====== PREDICTING'
isValidation = False
_, inputTable_test, outputTable_test, combinedTable_test = condense_data('test_v2', DATA_DIR, isTraining=isValidation,
                                                                         readFromFiles=True, outputDir=CONDENSED_TABLES_DIR)
pdf(inputTable_test)
X_test = Normalizer().fit_transform(Imputer().fit_transform(inputTable_test))

preds = combinedClf.predict(X_test)

if isValidation:
    combinedTable_test['PRED'] = preds
    combinedTable_test.to_csv(DATA_DIR + 'val.csv')

# --- write out to file
res = pandas.DataFrame(index = inputTable_test.index, data = preds)
res.columns = ['plan']
res.to_csv('/home/jj/code/Kaggle/allstate/submissions/gbc_indiv.csv')