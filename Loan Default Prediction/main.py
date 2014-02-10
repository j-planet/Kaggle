from datetime import datetime
from pprint import pprint
import numpy as np
from sklearn.ensemble.tests.test_gradient_boosting import test_check_max_features

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV

from Kaggle.utilities import makePipe, Normalizer
from helpers import *
from globalVars import *


# ---------- read in data
smallTrainX, smallTrainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modSmallTrain.csv",
                                        selectFeatures=False, enc=None)

# # binaryTrainY, regTrainX, regTrainY, _ = split_class_reg(smallTrainX, smallTrainY)
#
# # ---------- select learner
# imputerToTry = ('filler', (Imputer(), {'strategy': ['mean', 'median', 'most_frequent']}))
# normalizerToTry = ('normalizer', (Normalizer(), {'method': ['standardize', 'rescale']}))
imputerToTry = ('filler', (Imputer(strategy='mean'), {}))
normalizerToTry = ('normalizer', (Normalizer(), {}))
pcaReducerToTry = ('PCAer', (PcaEr(total_var=0.85), {}))
rfReducerToTry = ('RFer', (RandomForester(num_features=25, n_estimators=25),{}))
classifierToTry = ('GBR', (GradientBoostingRegressor(loss='lad', n_estimators=5, max_depth=3, subsample=0.7, learning_rate=0.1),
                           {'max_features': ['auto', 'sqrt', 'log2']}))
# classifierToTry = ('GBR', (GradientBoostingRegressor(loss='lad'), {'n_estimators': [5, 20]}) )
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

# # ---------- learn the full training data
fullTrainX, fullTrainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTrain.csv",
                                           selectFeatures=False, enc=None)

bestPipe = gscv.best_estimator_

# bestPipe = Pipeline(steps=[('filler', Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)),
#                            ('normalizer', Normalizer(method='standardize')),
#                            ('PCAer', PcaEr(fixed_num_components=None, method='PCA', total_var=0.85)),
#                            ('RFer', RandomForester(max_depth=None, min_samples_split=2, n_estimators=25, num_features=25)),
#                            ('GBR', GradientBoostingRegressor(loss='lad', n_estimators=5, max_depth=3, subsample=0.7,
#                                                              learning_rate=0.1, max_features='auto'))])

print 'jj score:', quick_score(bestPipe, smallTrainX, smallTrainY)

bestPipe.fit(fullTrainX, fullTrainY)

# ----------  predict & write to file
testData, _, testDataIds, _ = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTest.csv",
                                     selectFeatures=False, enc=enc)
pred = bestPipe.predict(testData)
write_predictions_to_file(testDataIds, pred, "/home/jj/code/Kaggle/Loan Default Prediction/submissions/GBR_cat.csv")

print '----- FIN -----'