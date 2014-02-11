from datetime import datetime
from pprint import pprint
import numpy as np
from sklearn.ensemble.tests.test_gradient_boosting import test_check_max_features

from sklearn.metrics import mean_absolute_error, zero_one
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

from Kaggle.utilities import makePipe, Normalizer
from helpers import *
from globalVars import *


# ---------- read in data
smallTrainX, smallTrainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modSmallTrain.csv",
                                             selectFeatures=False, enc=None)

binaryTrainY, regTrainX, regTrainY, _ = split_class_reg(smallTrainX, smallTrainY)

# ------ ==0 (binary) stuff ---------
# # ---------- select classifier
imputerToTry = ('filler', (Imputer(strategy='mean'), {}))
normalizerToTry = ('normalizer', (Normalizer(), {}))
pcaReducerToTry = ('PCAer', (PcaEr(total_var=0.85), {'whiten': [True, False]}))
rfReducerToTry = ('RFer', (RandomForester(num_features=0.5, n_estimators=5), {}))
classifierToTry = ('GBC', (GradientBoostingClassifier(subsample=0.7),
                           {'learning_rate': [0.05, 0.1], 'n_estimators': [25, 50]}))

pipe, allParamsDict = makePipe([imputerToTry, normalizerToTry, pcaReducerToTry, rfReducerToTry, classifierToTry])
gscv = GridSearchCV(pipe, allParamsDict, loss_func=zero_one, n_jobs=1, cv=4, verbose=5)

dt = datetime.now()
gscv.fit(smallTrainX, binaryTrainY)
print 'CV Took', datetime.now() - dt

print '\n>>> Grid scores:'
pprint(gscv.grid_scores_)
print '\n>>> Best Estimator:'
pprint(gscv.best_estimator_)
print '\n>>> Best score:', gscv.best_score_
print '\n>>> Best Params:'
pprint(gscv.best_params_)

bestClassifierPipe = gscv.best_estimator_
# ------ >0 (regression) stuff ---------
