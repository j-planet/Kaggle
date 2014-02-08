from datetime import datetime
from pprint import pprint
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV

from Kaggle.utilities import jjcross_val_score, makePipe, Normalizer
from helpers import *
from globalVars import *


# ---------- read in data
smallTrainX, smallTrainY, _ = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modSmallTrain.csv", False)


# fullTrainX, fullTrainY, _ = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTrain.csv", selectFeatures=False)
# binaryTrainY, regTrainX, regTrainY, _ = split_class_reg(smallTrainX, smallTrainY)

# ---------- select learner
# imputerToTry = ('filler', (Imputer(), {'strategy': ['mean', 'median', 'most_frequent']}))
# normalizerToTry = ('normalizer', (Normalizer(), {'method': ['standardize', 'rescale']}))
imputerToTry = ('filler', (Imputer(strategy='mean'), {}))
normalizerToTry = ('normalizer', (Normalizer(), {}))
pcaReducerToTry = ('PCAer', (PcaEr(total_var=0.999), {'total_var': [0.85, 0.95]}))
rfReducerToTry = ('RFer', (RandomForester(num_features=999, n_estimators=999), {'num_features': [0.5, 25], 'n_estimators': [25, 50]}))
# classifierToTry = ('GBR', (GradientBoostingRegressor(loss='lad'),
#                            {'n_estimators': [5, 7, 10, 20, 30, 50],
#                             'max_depth': [3, 5, 7, 10],
#                             'subsample': [0.7, 0.85, 1.0],
#                             'max_features': ['auto', 'sqrt', 'log2'],
#                             'learning_rate': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]}))
classifierToTry = ('GBR', (GradientBoostingRegressor(loss='lad'), {'n_estimators': [5, 20]}) )
pipe, allParamsDict = makePipe([imputerToTry, normalizerToTry, pcaReducerToTry, rfReducerToTry, classifierToTry])
gscv = GridSearchCV(pipe, allParamsDict, loss_func=mean_absolute_error, n_jobs=20, cv=4, verbose=5)
dt = datetime.now()
gscv.fit(smallTrainX, smallTrainY)
print 'Took', datetime.now() - dt

print '\n>>> Grid scores:'
pprint(gscv.grid_scores_)
print '\n>>> Best Estimator:'
pprint(gscv.best_estimator_)
print '\n>>> Best score:', gscv.best_score_
print '\n>>> Best Params:'
pprint(gscv.best_params_)
#
# # ---------- learn the full training data
# fullTrainX, fullTrainY, _ = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTrain.csv", False)
# dt = datetime.now()
# trainX, imp, scaler, pca, indices = select_features(smallTrainX, smallTrainY, pcaVarSum=0.9)
# print 'Took', datetime.now() - dt

# bestPipe = gscv.best_estimator_
# bestPipe = Pipeline(steps=[('filler', Imputer(axis=0, copy=True, missing_values='NaN', strategy='most_frequent',
#                                               verbose=0)),
#                            ('normalizer', Normalizer(method = 'standardize')),
#                            ('GBR', GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.5, loss='lad',
#                                                              max_depth=3, max_features='sqrt', min_samples_leaf=1,
#                                                              min_samples_split=2, n_estimators=5, random_state=None,
#                                                              subsample=0.7, verbose=0))])
# bestPipe = Pipeline(steps=[('GBR', GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.5, loss='lad',
#                                                              max_depth=3, max_features='sqrt', min_samples_leaf=1,
#                                                              min_samples_split=2, n_estimators=5, random_state=None,
#                                                              subsample=0.7, verbose=0))])
# bestPipe.fit(trainX, fullTrainY)
# print 'jj score:', jjcross_val_score(bestPipe, select_features(smallTrainX, smallTrainY, 0.95), smallTrainY, mean_absolute_error, 5, n_jobs=20).mean()

# ----------  predict & write to file
testData, _, testDataIds = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTest.csv", False)
testData = pca.transform(scaler.transform(imp.transform(testData)))[:, indices]
pred = bestPipe.predict(testData)
write_predictions_to_file(testDataIds, pred, "/home/jj/code/Kaggle/Loan Default Prediction/submissions/GBR_pca.csv")

print '----- FIN -----'