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
from pipes import *
from BinThenReg import *


smallTrainX, smallTrainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modSmallTrain.csv",
                                             selectFeatures=False, enc=None)

# ---------- select learner
pipe_prep, params_prep = prepPipes(simple=True)
pipe_class, params_class = classifierPipes(simple=True)
pipe_reg, params_reg = regressorPipes(simple=True)

# pipe = Pipeline(steps=pipe_prep.steps + pipe_reg.steps)
# allParamsDict = dict(params_prep.items() + params_reg.items())

pipe = BinThenReg(pipe_prep, pipe_class, pipe_reg)
params = BinThenReg.make_params_dict(params_prep, params_class, params_reg)

gscv = GridSearchCV(pipe, params, loss_func=mean_absolute_error, n_jobs=20, cv=4, verbose=5)
dt = datetime.now()
gscv.fit(smallTrainX, smallTrainY)
print 'Took', datetime.now() - dt

print_GSCV_info(gscv)

# ---------- double-check cv score
bestPipe = gscv.best_estimator_
dt = datetime.now()
print 'jj score:', quick_score(bestPipe, smallTrainX, smallTrainY)
print 'quick jj score took', datetime.now() - dt

# ---------- learn the full training data
fullTrainX, fullTrainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTrain.csv",
                                           selectFeatures=False, enc=None)

dt = datetime.now()
bestPipe.fit(fullTrainX, fullTrainY)
print 'Fitting to full training data took', datetime.now() - dt

# ----------  predict & write to file
write_predictions_to_file(predictor=bestPipe,
                          testDataFname="/home/jj/code/Kaggle/Loan Default Prediction/Data/modTest.csv",
                          enc=enc,
                          outputFname="/home/jj/code/Kaggle/Loan Default Prediction/submissions/GBR_binReg.csv")

print '----- FIN -----'