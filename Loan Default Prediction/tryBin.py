from datetime import datetime
import numpy as np
from copy import deepcopy

from sklearn.ensemble.tests.test_gradient_boosting import test_check_max_features
from sklearn.metrics import mean_absolute_error, zero_one
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

from Kaggle.utilities import makePipe, Normalizer
from helpers import *
from globalVars import *


# ======== read in data
smallTrainX, smallTrainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modSmallTrain.csv",
                                             selectFeatures=False, enc=None)
binaryTrainY_small, _, regTrainY_small, nonZeroMask_small = split_class_reg(smallTrainX, smallTrainY)

# ======== ==0 (binary) stuff ========
imputerToTry = ('filler', (Imputer(strategy='mean'), {}))
normalizerToTry = ('normalizer', (Normalizer(), {}))
pcaReducerToTry_class = ('PCAer', (PcaEr(total_var=0.85), {'whiten': [True, False]}))
rfReducerToTry_class = ('RFer', (RandomForester(num_features=0.5, n_estimators=5), {}))
classifierToTry = ('GBC', (GradientBoostingClassifier(subsample=0.7),
                           {'learning_rate': [0.05, 0.1], 'n_estimators': [25, 50]}))

pipe_classify, allParamsDict_classify = makePipe([imputerToTry, normalizerToTry, pcaReducerToTry_class, rfReducerToTry_class, classifierToTry])

# gscv_classify = GridSearchCV(pipe_classify, allParamsDict_classify, n_jobs=20, cv=4, verbose=5)
# dt = datetime.now()
# gscv_classify.fit(smallTrainX, binaryTrainY_small)    # use binary y
# print 'CV Took', datetime.now() - dt
# print_GSCV_info(gscv_classify)


# bestClassifierPipe = gscv_classify.best_estimator_
bestClassifierPipe = deepcopy(pipe_classify)
bestClassifierPipe.set_params(**{'GBC__learning_rate': 0.05, 'GBC__n_estimators': 25, 'PCAer__whiten': True})

# ======== >0 (regression) stuff ========
# imputerToTry = ('filler', (Imputer(strategy='mean'), {}))
# normalizerToTry = ('normalizer', (Normalizer(), {}))
newx = bestClassifierPipe.named_steps['normalizer'].fit_transform(bestClassifierPipe.named_steps['filler'].fit_transform(smallTrainX))

pcaReducerToTry_reg = ('PCAer', (PcaEr(total_var=0.85), {'whiten': [True, False]}))
rfReducerToTry_reg = ('RFer', (RandomForester(num_features=0.5, n_estimators=5), {}))
regressorToTry = ('GBR', (GradientBoostingRegressor(loss='lad', max_features='auto', learning_rate=0.1, n_estimators=5),
                          {'subsample': [0.5, 0.7, 1]}))
pipe_regress, allParamsDict_regress = makePipe([pcaReducerToTry_reg, rfReducerToTry_reg, regressorToTry])

gscv_regress = GridSearchCV(pipe_regress, allParamsDict_regress, loss_func=mean_absolute_error, n_jobs=20, cv=4, verbose=5)
dt = datetime.now()
gscv_regress.fit(newx[nonZeroMask_small], smallTrainY[nonZeroMask_small])      # use x and y where y>0
print 'Took', datetime.now() - dt
print_GSCV_info(gscv_regress)


bestRegPipe = gscv_regress.best_estimator_
# bestRegPipe = deepcopy(pipe_regress)
# bestRegPipe.set_params(**{'GBR__subsample': 1, 'PCAer__whiten': False})


# ======== learn the full data ========
fullTrainX, fullTrainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTrain.csv",
                                           selectFeatures=False, enc=None)
binaryTrainY, regTrainX, regTrainY, _ = split_class_reg(fullTrainX, fullTrainY)
bestClassifierPipe.fit(fullTrainX, binaryTrainY)
bestRegPipe.fit(regTrainX, regTrainY)

# ======== predict & write to file ========
testData, _, testDataIds, _ = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTest.csv",
                                        selectFeatures=False, enc=enc)