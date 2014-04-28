import sys
sys.path.append('/home/jj/code/Kaggle/allstate')

from Kaggle.utilities import print_missing_values_info, jjcross_val_score

from sklearn.preprocessing import Imputer, Normalizer, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

from globalVars import *
from helpers import *



# ======= read data =======
partname = 'train'
# origDataFpath = os.path.join(os.path.dirname(__file__), 'Data', partname + '.csv')
# outputFpath = os.path.join(os.path.dirname(__file__), 'Data', 'condensed_' + partname + '.csv')
origDataFpath = DATA_PATH + partname + '.csv'
outputFpath = DATA_PATH + 'condensed_' + partname + '.csv'

_, inputTable, outputTable, _ = condense_data(origDataFpath, isTraining=True,
                                                                    outputFpath = outputFpath, verbose=False)
X_train = Normalizer().fit_transform(Imputer().fit_transform(inputTable))  # TODO: better imputation
y_train = CombinedClassifier.combine_outputs(np.array(outputTable))

pdf(inputTable)

# ======= validate classifiers =======
# plot_feature_importances(X_train, outputTable, inputTable.columns)

# print '====== combined accuracy score'
#
# combinedClf = CombinedClassifier.create_by_cloning(
#     GradientBoostingClassifier(subsample=0.7, n_estimators=50, learning_rate=0.1), len(OUTPUT_COLS))
#
# print jjcross_val_score(combinedClf, X_train, y_train, accuracy_score, cv=5, n_jobs=20)


# print '====== individual accuracy score'
# for col in outputTable.columns:
#     y = outputTable[col]
#     clf = GradientBoostingClassifier(subsample=0.8, n_estimators=50, learning_rate=0.05)
#     # clf = RandomForestClassifier(n_estimators=25, n_jobs=20)



#
#     print col, jjcross_val_score(clf, X_train, y, accuracy_score, cv=5, n_jobs=20).mean()

# ======= predict =======
partname = 'test_v2'
isValidation = False
testFpath = DATA_PATH + partname + '.csv'

_, inputTable_test, outputTable_test, combinedTable_test = condense_data(testFpath, isTraining=isValidation, verbose=False)
X_test = Normalizer().fit_transform(Imputer().fit_transform(inputTable_test))

print '====== TRAINING'

# combinedClf = CombinedClassifier.create_by_cloning(
#     GradientBoostingClassifier(subsample=0.8, n_estimators=25, learning_rate=0.05), len(OUTPUT_COLS))
combinedClf = CombinedClassifier.create_by_cloning(
    RandomForestClassifier(n_estimators=50, n_jobs=20), len(OUTPUT_COLS))
combinedClf.fit(X_train, y_train)

print '====== PREDICTING'
preds = combinedClf.predict(X_test)

if isValidation:
    combinedTable_test['PRED'] = preds
    combinedTable_test.to_csv(DATA_PATH + 'val.csv')

# --- write out to file
res = pandas.DataFrame(index = inputTable_test.index, data = preds)
res.columns = ['plan']
res.to_csv('/home/jj/code/Kaggle/allstate/submissions/rf.csv')