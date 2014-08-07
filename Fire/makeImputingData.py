# Calculate a constant value used to replace the missing values for each field

import os
import pandas
import numpy as np

from utilities import mode
from globalVars import *


def make_imputer(trainFname, testFname, outputDir):
    """
    makes an imputer based on both train and test data
    :param trainFname:
    :param testFname:
    :param imputerStrategy: one of "mean", "median" or "most_frequent"
    :return: fitted imputer
    """

    x_train = pandas.read_csv(trainFname)
    x_test = pandas.read_csv(testFname)
    x_full = pandas.concat([x_train, x_test])

    res_mean = {}
    res_median = {}
    res_mode = {}

    # set Z as NA
    x_full = x_full.replace(to_replace='Z', value=np.nan)

    # convert discrete values to numeric values
    for col in DISCRETE_COLS:
        for oldVal, newVal in DISCRETE_COLS_LOOKUP[col].iteritems():
            x_full[col] = x_full[col].replace(to_replace=oldVal, value=newVal)

    # calculate stats
    for col in x_full.columns:
        if col in NON_PREDICTOR_COLS:
            continue

        curData = np.array(x_full[col], dtype=np.float)
        curData = curData[np.logical_not(np.isnan(curData))]

        res_mean[col] = [np.mean(curData)]
        res_median[col] = [np.median(curData)]
        res_mode[col] = [mode(curData)]


    pandas.DataFrame(res_mean).to_csv(os.path.join(outputDir, 'impute_mean.csv'), index=False)
    pandas.DataFrame(res_median).to_csv(os.path.join(outputDir, 'impute_median.csv'), index=False)
    pandas.DataFrame(res_mode).to_csv(os.path.join(outputDir, 'impute_mode.csv'), index=False)


if __name__=='__main__':
    make_imputer('/home/jj/code/Kaggle/Fire/Data/train.csv',
                 '/home/jj/code/Kaggle/Fire/Data/test.csv',
                 '/home/jj/code/Kaggle/Fire')