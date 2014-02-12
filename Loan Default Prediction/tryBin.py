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


class BinThenReg(object):
    """
    first classify target values as 0 or positive, then predict the positive values
    """

    def __init__(self, common_preprocessing_pipe, classifier_pipe, regressor_pipe):
        """
        @param common_preprocessing_pipe: the preprocessing step done prior to BOTH classifying and regressing
        """

        self.common_preprocessing_pipe = deepcopy(common_preprocessing_pipe)
        self.classifier_pipe = deepcopy(classifier_pipe)
        self.regressor_pipe = deepcopy(regressor_pipe)

    def set_params(self, **params):

        self.common_preprocessing_pipe.set_params(dict((name[(len('common_preprocessing_pipe'))+2:], v)
                                                        for name, v in params.iteritems()
                                                        if 'common_preprocessing_pipe' in name))

        self.classifier_pipe.set_params(dict((name[(len('classifier_pipe'))+2:], v)
                                             for name, v in params.iteritems()
                                             if 'classifier_pipe' in name))

        self.regressor_pipe.set_params(dict((name[(len('regressor_pipe'))+2:], v)
                                            for name, v in params.iteritems()
                                            if 'regressor_pipe' in name))

        return self

    def fit(self, X, y):

        binaryY, _, regY, nonZeroMask = split_class_reg(X, y)

        # apply common_preprocessing_pipe
        newX = self.common_preprocessing_pipe.fit_transform(X)

        # apply classifier_pipe
        self.classifier_pipe.fit(newX, binaryY)

        # apply regressor_pipe
        self.regressor_pipe.fit(newX[nonZeroMask], regY)

    def predict(self, X):

        # apply common_preprocessing_pipe
        newX = self.common_preprocessing_pipe.fit_transform(X)  # just transform?

        # apply classifier_pipe
        binaryOutput = self.classifier_pipe.predict(newX)
        nonZeroMask = binaryOutput > 0

        # apply regressor_pipe
        regOutput = self.regressor_pipe.predict(newX[nonZeroMask])

        # combine binary and regression output
        binaryOutput[nonZeroMask] = regOutput
        binaryOutput[(binaryOutput>0) & (binaryOutput<1)] = 1   # set 0.sth to 1

    @staticmethod
    def make_params_dict(prepParamsDict, classParamsDict, regParamsDict):
        """
        @return: a dictionary
        """

        d1 = dict(('common_preprocessing_pipe__' + k, v) for k, v in prepParamsDict.iteritems())
        d2 = dict(('classifier_pipe__' + k, v) for k, v in classParamsDict.iteritems())
        d3 = dict(('regressor_pipe__' + k, v) for k, v in regParamsDict.iteritems())

        return dict(d1.items() + d2.items() + d3.items())

# ======== read in data
smallTrainX, smallTrainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modSmallTrain.csv",
                                             selectFeatures=False, enc=None)
binaryTrainY_small, _, regTrainY_small, nonZeroMask_small = split_class_reg(smallTrainX, smallTrainY)

pipe_prep, params_prep = prepPipes(simple=True)
pipe_class, params_class = classifierPipes(simple=True)
pipe_reg, params_reg = regressorPipes(simple=True)

BinThenReg(pipe_prep, pipe_class, pipe_reg)

# ======== ==0 (binary) stuff ========
imputerToTry = ('filler', (Imputer(strategy='mean'), {}))
normalizerToTry = ('normalizer', (Normalizer(), {}))
# pcaReducerToTry_class = ('PCAer', (PcaEr(total_var=0.85), {'whiten': [True, False]}))
rfReducerToTry_class = ('RFer', (RandomForester(n_estimators=5, num_features=99), {'num_features':[15, 25]}))
classifierToTry = ('GBC', (GradientBoostingClassifier(subsample=0.7),
                           {'learning_rate': [0.05, 0.1], 'n_estimators': [25, 50]}))

# pipe_classify, allParamsDict_classify = makePipe([imputerToTry, normalizerToTry, pcaReducerToTry_class, rfReducerToTry_class, classifierToTry])
pipe_classify, allParamsDict_classify = makePipe([imputerToTry, normalizerToTry, rfReducerToTry_class, classifierToTry])

gscv_classify = GridSearchCV(pipe_classify, allParamsDict_classify, n_jobs=20, cv=4, verbose=5)
dt = datetime.now()
gscv_classify.fit(smallTrainX, binaryTrainY_small)    # use binary y
print 'CV Took', datetime.now() - dt
print_GSCV_info(gscv_classify)


bestClassifierPipe = gscv_classify.best_estimator_
# bestClassifierPipe = deepcopy(pipe_classify)
# bestClassifierPipe.set_params(**{'GBC__learning_rate': 0.05, 'GBC__n_estimators': 25})

# ======== >0 (regression) stuff ========
# imputerToTry = ('filler', (Imputer(strategy='mean'), {}))
# normalizerToTry = ('normalizer', (Normalizer(), {}))

newx = bestClassifierPipe.named_steps['normalizer'].fit_transform(bestClassifierPipe.named_steps['filler'].fit_transform(smallTrainX))

# pcaReducerToTry_reg = ('PCAer', (PcaEr(total_var=0.85), {'whiten': [True, False]}))
rfReducerToTry_reg = ('RFer', (RandomForester(n_estimators=5, num_features=99), {'num_features': [15, 25]}))
regressorToTry = ('GBR', (GradientBoostingRegressor(loss='lad', max_features='auto', learning_rate=0.1, n_estimators=5),
                          {'subsample': [0.5, 0.7, 1]}))
# pipe_regress, allParamsDict_regress = makePipe([pcaReducerToTry_reg, rfReducerToTry_reg, regressorToTry])
pipe_regress, allParamsDict_regress = makePipe([rfReducerToTry_reg, regressorToTry])

gscv_regress = GridSearchCV(pipe_regress, allParamsDict_regress, loss_func=mean_absolute_error, n_jobs=20, cv=4, verbose=5)
dt = datetime.now()
gscv_regress.fit(newx[nonZeroMask_small], smallTrainY[nonZeroMask_small])      # use x and y where y>0
print 'Took', datetime.now() - dt
print_GSCV_info(gscv_regress)


bestRegPipe = gscv_regress.best_estimator_
# bestRegPipe = deepcopy(pipe_regress)
# bestRegPipe.set_params(**{'GBR__subsample': 0.7})


# ======== learn the full data ========
fullTrainX, fullTrainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTrain.csv",
                                           selectFeatures=False, enc=None)
binaryTrainY, _, regTrainY, nonZeroMask = split_class_reg(fullTrainX, fullTrainY)

dt = datetime.now()
bestClassifierPipe.fit(fullTrainX, binaryTrainY)
newx = bestClassifierPipe.named_steps['normalizer'].fit_transform(bestClassifierPipe.named_steps['filler'].fit_transform(fullTrainX))
bestRegPipe.fit(newx[nonZeroMask], regTrainY)
print 'Fitting to full training data took', datetime.now() - dt

# # ======== predict & write to file ========
testData, _, testDataIds, _ = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTest.csv",
                                        selectFeatures=False, enc=enc)
dt = datetime.now()
binaryOutput = bestClassifierPipe.predict(testData)
nonZeroMaskOutput = binaryOutput > 0
newx = bestClassifierPipe.named_steps['normalizer'].fit_transform(bestClassifierPipe.named_steps['filler'].fit_transform(testData))
regOutput = bestRegPipe.predict(newx[nonZeroMaskOutput])
binaryOutput[nonZeroMaskOutput] = regOutput     # final output
binaryOutput[(binaryOutput>0) & (binaryOutput<1)] = 1   # set 0.sth to 1
print 'predicting took', datetime.now() - dt

write_predictions_to_file(testDataIds, binaryOutput, "/home/jj/code/Kaggle/Loan Default Prediction/submissions/classReg_nopca.csv")