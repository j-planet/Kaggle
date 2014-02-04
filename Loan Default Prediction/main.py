import pandas
from datetime import datetime
import csv
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV

from Kaggle.utilities import jjcross_val_score, makePipe


def select_features(xData, yData, mandatoryColumns=None):
    """
    Feature selection
    @param xData: a pandas data frame
    @param yData: a vector
    @param mandatoryColumns: a list of features that must be selected
    @return:
        selected xData (pandas data frame)
        columns selected (list)
    """

    # F-score
    sp = SelectPercentile(score_func=f_regression, percentile=10)
    sp.fit(xData, yData)
    mask = sp.get_support()
    filteredXData_fReg = xData[xData.columns[mask]]
    print "f_regression selected:", filteredXData_fReg.columns

    # Random Forest
    numFeatures_rf = int(len(filteredXData_fReg.columns) * 0.5)
    forest = ExtraTreesRegressor(n_estimators=100)
    forest.fit(filteredXData_fReg, yData)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1][:numFeatures_rf]
    columns_rf = filteredXData_fReg.columns[indices]

    print "Random Forest feature ranking selected features:", columns_rf

    # some rf plotting
    plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(indices)), columns_rf)
    plt.show()

    columns_final = set(columns_rf).union(mandatoryColumns)
    print "Final features selected:", columns_final

    return xData[columns_final], columns_final


def print_missing_values_info(data):
    """
    Prints the number of missing data columns and values of a pandas data frame.
    @param data 2D pandas data frame
    @return None
    """

    temp_col = pandas.isnull(data).sum()
    temp_row = pandas.isnull(data).sum(axis=1)

    print 'The data has', (temp_col > 0).sum(), 'or', round(100. * (temp_col > 0).sum() / data.shape[1], 1), '% columns with missing values.'
    print 'The data has', (temp_row > 0).sum(), 'or', round(100. * (temp_row > 0).sum() / data.shape[0], 1), '% rows with missing values.'

    print 'The data has', temp_col.sum(), 'or', round(
        100. * temp_col.sum() / (data.shape[0] * data.shape[1]), 1), '% missing values.'


def set_vars_as_type(df, varNames, dtype):
    """
    Set certain variables of a pandas df to a pre-defined data type. Changes df in place.
    @param df pandas data frame
    @param varNames list of strings
    @param dtype data type
    """

    myVars = list(set(df.columns).intersection(set(varNames)))
    df[myVars] = df[myVars].astype(dtype)


def impute_data(xData, yData):
    """
    Filles in missing values
    @param xData: a pandas data frame of x values
    @param yData: a vector of y values
    @return: new xdata (a pandas df)
        the imputer that can be called to transform future data
    """

    imp = Imputer(strategy='mean')
    imp.fit(xData, yData)
    newXData = pandas.DataFrame(imp.transform(xData), columns=xData.columns)

    return newXData, imp

mandatoryColumns = ['f11', 'f12', 'f462', 'f463', 'f473', 'f474', 'f602', 'f603', 'f605', 'f776', 'f777', 'f778', 'f2',
                    'f4', 'f5', 'f6']

# ---------- read in data
trainFname = "/home/jj/code/Kaggle/Loan Default Prediction/Data/modSmallTrain.csv"
# trainFname = "/home/jj/code/Kaggle/Loan Default Prediction/Data/modTrain.csv"
trainData_orig = pandas.read_csv(trainFname)

# remove unused columns
if 'Unnamed: 0' in trainData_orig.columns: del trainData_orig['Unnamed: 0']
del trainData_orig['id']

# set discrete variables
discreteVars = ['f776', 'f777', 'f778', 'f6']
trainData_orig[discreteVars] = trainData_orig[discreteVars].astype(object)


# separate into X & y values
xData = trainData_orig[[col for col in trainData_orig.columns if not col=='loss']]
yVec = trainData_orig.loss

# ---------- handle missing values (replace with mean for now)
print 'Training:'
print_missing_values_info(trainData_orig)
set_vars_as_type(xData, discreteVars, object)


# ---------- feature selection
# filteredXData_final, columns_final = selectFeatures(xData, yVec, mandatoryColumns)
columns_final = ['f536', 'f602', 'f603', 'f4', 'f605', 'f6', 'f2', 'f696', 'f473', 'f344', 'f261', 'f767', 'f285', 'f765', 'f666',
                 'f281', 'f282', 'f665', 'f221', 'f323', 'f322', 'f47', 'f5', 'f103', 'f667', 'f68', 'f67', 'f474', 'f675', 'f674',
                 'f676', 'f631', 'f462', 'f468', 'f425', 'f400', 'f778', 'f405', 'f776', 'f463', 'f428', 'f471', 'f777', 'f314', 'f211',
                 'f315', 'f252', 'f251', 'f426', 'f12', 'f11', 'f70']
filteredXData_final = xData[list(columns_final)]

# ---------- learn
imputerToTry = ('filler', (Imputer(), {'strategy': ['mean', 'median', 'most_frequent']}))
classifierToTry = ('GBR', (GradientBoostingRegressor(loss='lad'),
                           {'n_estimators': [5, 10, 20, 25, 30, 50, 100],
                            'max_depth': [3, 5, 7, 10],
                            'subsample': [0.7, 0.85, 1.0]}))
# classifierToTry = ('GBR', (GradientBoostingRegressor(loss='lad'), {'n_estimators': [5, 20], 'max_depth': [3, 10]}) )
pipe, allParamsDict = makePipe([imputerToTry, classifierToTry])
gscv = GridSearchCV(pipe, allParamsDict, loss_func=mean_absolute_error, n_jobs=20, cv=5, verbose=5)
dt = datetime.now()
gscv.fit(filteredXData_final, yVec)
print 'Took', datetime.now() - dt

print '\n>>> Grid scores:'
pprint(gscv.grid_scores_)
print '\n>>> Best Estimator:'
pprint(gscv.best_estimator_)
print '\n>>> Best score:', gscv.best_score_
print '\n>>> Best Params:'
pprint(gscv.best_params_)


# print '\n---------- learn'
# pipe.set_params(filler__strategy='mean', GBR__n_estimators=5, GBR__max_depth=3)
# print jjcross_val_score(pipe, filteredXData_final, yVec, mean_absolute_error, cv=10, n_jobs=10).mean()


# ----------  predict
# testFname = "/home/jj/code/Kaggle/Loan Default Prediction/Data/modTest.csv"
# testData_orig = pandas.read_csv(testFname)
# testDataIds = testData_orig.id
#
# if 'Unnamed: 0' in testData_orig.columns: del testData_orig['Unnamed: 0']
# del testData_orig['id']
# testData_orig[discreteVars] = testData_orig[discreteVars].astype(object)
#
# print 'Testing:'
# print_missing_values_info(testData_orig)


# testData = pandas.DataFrame(imp.fit_transform(testData_orig), columns=testData_orig.columns)     # impute
# # testData = pandas.DataFrame(imp.transform(testData_orig), columns=testData_orig.columns)     # impute
# testData = testData[columns_final]                 # select features to use
# set_vars_as_type(testData, discreteVars, object)
# pred = clf.predict(testData)                    # predict
#
#
#
# # ---------- write predictions to file
# output = np.transpose(np.vstack((testDataIds, pred.round().astype(int))))
#
# outputFname = "/home/jj/code/Kaggle/Loan Default Prediction/submissions/gradientBoostingReg_meanImpTest.csv"
# with open(outputFname, 'wb') as outputFile:
#     writer = csv.writer(outputFile)
#     writer.writerow(['id', 'loss'])
#     writer.writerows(output)
#
#
# plt.figure()
# plt.hist(yVec, bins=20, color='green', alpha=0.5)
# plt.hist(output[:, 1], bins=20, color='red', alpha=0.5)
# plt.show()
#
#
# from sklearn import tree
# clf2 = tree.DecisionTreeRegressor()
# clf2.fit(trainData_orig[['f1', 'f776', 'f777', 'f778']], trainData_orig.loss)
#
# zero = trainData_orig[trainData_orig.f776==0]
# one = trainData_orig[trainData_orig.f776==1]
# sum(zero.loss==0)*1./zero.shape[0]
# sum(one.loss==0)*1./one.shape[0]
# zero.loss[zero.loss>0].mean()
# one.loss[one.loss>0].mean()
# zero.loss.mean()
# one.loss.mean()

print '----- FIN -----'