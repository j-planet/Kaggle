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
# smallTrainX, smallTrainY, _ = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modSmallTrain.csv", selectFeatures=False)
fullTrainX, fullTrainY, _ = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTrain.csv", selectFeatures=False)
# binaryTrainY, regTrainX, regTrainY, _ = split_class_reg(smallTrainX, smallTrainY)

select_features(fullTrainX, fullTrainY, mandatoryColumns)

# ---------- select learner
# imputerToTry = ('filler', (Imputer(), {'strategy': ['mean', 'median', 'most_frequent']}))
# normalizerToTry = ('normalizer', (Normalizer(), {'method':['standardize', 'rescale']}))
# classifierToTry = ('GBR', (GradientBoostingRegressor(loss='lad'),
#                            {'n_estimators': [5, 7, 10, 20, 30, 50],
#                             'max_depth': [3, 5, 7, 10],
#                             'subsample': [0.7, 0.85, 1.0],
#                             'max_features': ['auto', 'sqrt', 'log2'],
#                             'learning_rate': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]}))
# # classifierToTry = ('GBR', (GradientBoostingRegressor(loss='lad'), {'n_estimators': [5, 20], 'max_depth': [3, 10]}) )
# pipe, allParamsDict = makePipe([imputerToTry, normalizerToTry, classifierToTry])
# gscv = GridSearchCV(pipe, allParamsDict, loss_func=mean_absolute_error, n_jobs=15, cv=5, verbose=5)
# dt = datetime.now()
# gscv.fit(smallTrainX, smallTrainY)
# print 'Took', datetime.now() - dt
#
# print '\n>>> Grid scores:'
# pprint(gscv.grid_scores_)
# print '\n>>> Best Estimator:'
# pprint(gscv.best_estimator_)
# print '\n>>> Best score:', gscv.best_score_
# print '\n>>> Best Params:'
# pprint(gscv.best_params_)
#
# # ---------- learn the full training data
# fullTrainX, fullTrainY, _ = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTrain.csv")
# bestPipe = gscv.best_estimator_
# # bestPipe = Pipeline(steps=[('filler', Imputer(axis=0, copy=True, missing_values='NaN', strategy='most_frequent',
# #                                               verbose=0)),
# #                            ('normalizer', Normalizer(method = 'standardize')),
# #                            ('GBR', GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.5, loss='lad',
# #                                                              max_depth=3, max_features='sqrt', min_samples_leaf=1,
# #                                                              min_samples_split=2, n_estimators=5, random_state=None,
# #                                                              subsample=0.7, verbose=0))])
# bestPipe.fit(fullTrainX, fullTrainY)
# print 'jj score:', jjcross_val_score(bestPipe, smallTrainX, smallTrainY, mean_absolute_error, 5, n_jobs=20).mean()
#
# # ----------  predict & write to file
# testData, _, testDataIds = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTest.csv")
# pred = bestPipe.predict(testData)
# write_predictions_to_file(testDataIds, pred, "/home/jj/code/Kaggle/Loan Default Prediction/submissions/gridSearchCV_GBR_standardized.csv")

print '----- FIN -----'