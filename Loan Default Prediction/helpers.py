import numpy as np
import pandas
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.ensemble import ExtraTreesRegressor

from globalVars import *


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


def make_data(dataFname):
    """
    reads x and y data (no imputation, yes feature selection)
    @param dataFname: name of the training csv file
    @return xdata, ydata (None if test data), ids
    """

    origData = pandas.read_csv(dataFname)
    ids = origData['id']

    # remove unused columns
    if 'Unnamed: 0' in origData.columns: del origData['Unnamed: 0']
    del origData['id']

    # separate into X & y values
    xData = origData[[col for col in origData.columns if not col=='loss']]
    set_vars_as_type(xData, discreteVars, object)
    yVec = origData.loss if 'loss' in origData.columns else None

    print_missing_values_info(origData)

    # feature selection
    # filteredXData_final, columns_final = selectFeatures(xData, yVec, mandatoryColumns)
    columns_final = ['f536', 'f602', 'f603', 'f4', 'f605', 'f6', 'f2', 'f696', 'f473', 'f344', 'f261', 'f767', 'f285', 'f765', 'f666',
                     'f281', 'f282', 'f665', 'f221', 'f323', 'f322', 'f47', 'f5', 'f103', 'f667', 'f68', 'f67', 'f474', 'f675', 'f674',
                     'f676', 'f631', 'f462', 'f468', 'f425', 'f400', 'f778', 'f405', 'f776', 'f463', 'f428', 'f471', 'f777', 'f314', 'f211',
                     'f315', 'f252', 'f251', 'f426', 'f12', 'f11', 'f70']
    # columns_final = xData.columns       # use ALL features
    filteredXData = xData[list(columns_final)]

    return filteredXData, yVec, ids


def write_predictions_to_file(ids, predictions, outputFname):
    """
    write output to file
    """

    featureSelectionOutput = np.transpose(np.vstack((ids, predictions.round().astype(int))))

    with open(outputFname, 'wb') as outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(['id', 'loss'])
        writer.writerows(featureSelectionOutput)

