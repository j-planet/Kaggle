__author__ = 'jennyyuejin'

import sys
sys.path.append('/Users/JennyYueJin/K/NDSB')

from global_vars import DATA_DIR, CLASS_MAPPING, CLASS_NAMES
from Utilities.utilities import plot_feature_importances

import subprocess
import datetime
import itertools
import glob
import os
from pprint import pprint
from multiprocessing import cpu_count

from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
import numpy as np
import pandas
import scipy.stats as stats
import seaborn as sns
from skimage.feature import peak_local_max

from fileMangling import read_test_data_given_path, write_train_data_to_files, write_test_data_to_files, make_submission_file
from features import FEATURE_NAMES
plt.ioff()


def print_matrix(mat, axis):
    assert axis in [0, 1]

    for i in range(mat.shape[axis]):
        if axis==0:
            print 'Row', i
            print mat[i, :]
        else:       # axis can only be 1 now
            print 'Column', i
            print mat[:,i]


# find the largest nonzero region
def get_largest_region(props, labelmap, imagethres):
    areas = [None
             if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50
             else regionprop.filled_area
             for regionprop in props]

    return props[np.argmax(areas)] if len(areas) > 0 else None


def plot_ratio_distns_for_pairs(classNames, minimumSize=20):
    # Loop through the classes two at a time and compare their distributions of the Width/Length Ratio

    #Create a DataFrame object to make subsetting the data on the class
    df = pandas.DataFrame({"class": y[:], "ratio": X_train[:, X_train.shape[1] - 1]})
    df = df[df['ratio'] > 0]    # suppress zeros

    # choose a few large classes to better highlight the distributions
    counts = df["class"].value_counts()
    largeclasses = np.array(counts[counts > minimumSize].index, dtype=int)

    plt.figure(figsize=(60, 40))
    bins = [x*0.01 for x in range(100)]

    # Loop through 20 of the classes
    for j in range(0, 20, 2):

        subfig = plt.subplot(2, 5, j/2 + 1)

        # Plot the normalized histograms for two classes
        classind1 = largeclasses[j]
        classind2 = largeclasses[j+1]

        plt.hist(df[df["class"] == classind1]["ratio"].values,
                 alpha=0.5, bins=bins,
                 label=classNames[classind1], normed=1)

        plt.hist(df[df["class"] == classind2]["ratio"].values,
                 alpha=0.5, bins=bins, label=classNames[classind2], normed=1)

        subfig.set_ylim([0., 10.])

        plt.legend(loc='upper right')
        plt.xlabel("Width/Length Ratio")

    plt.show()


def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss


def predict_and_submit(X_train, y_train, testFpath, clfKlass, **clfArgs):

    X_test, testFnames = read_test_data_given_path(testFpath)

    clf = clfKlass(**clfArgs)
    clf.fit(X_train, y_train)

    pred = np.zeros((X_test.shape[0], len(CLASS_NAMES)))
    pred[:, np.sort(list(set(y_train)))] = clf.predict_proba(X_test)

    make_submission_file(pred, testFnames, fNameSuffix='nonDN')


def evaluate(X, y, clfKlass, **clfArgs):
    """
    produces CV accuracy score, classification report by class and MCLL (multi-class log loss)
    :param X:
    :param y:
    :param clfKlass:
    :param clfArgs:
    :return: MCLL
    """

    print '======= Evaluating ======='

    # # print "CV-ing"
    # scores = cross_validation.cross_val_score(RandomForestClassifier(n_estimators=100, n_jobs=cpu_count()-1),
    #                                           X_train, y, cv=5, n_jobs=cpu_count()-1)
    #
    # print "Accuracy of all classes:", np.mean(scores)


    # Get the probability predictions for computing the log-loss function
    # prediction probabilities number of samples, by number of classes
    y_pred = y * 0
    y_pred_mat = np.zeros((len(y), len(CLASS_NAMES)))   # forcing all class names, for testing with partial data


    for trainInd, testInd in KFold(y, n_folds=5):
        clf = clfKlass(**clfArgs)
        clf.fit(X[trainInd, :], y[trainInd])

        y_pred[testInd] = clf.predict(X[testInd, :])

        set_array_vals(y_pred_mat, testInd, np.sort(list(set(y))), clf.predict_proba(X[testInd, :]))


    # print '>>>>>> Classification Report'
    # print classification_report(y, y_pred, target_names=CLASS_NAMES)

    MCLL = multiclass_log_loss(y, y_pred_mat)
    print '\n>>>>>>>Multi-class Log Loss =', MCLL

    return MCLL


def set_array_vals(target, rowInds, colInds, source):

    assert len(rowInds), len(colInds) == source.shape

    wideMat = np.zeros((len(rowInds), target.shape[1]))
    wideMat[:, colInds] = source

    target[rowInds, :] = wideMat


def plot_pixel_importances(width, height, X, y, **rfArgs):
    clf = RandomForestClassifier(n_jobs=cpu_count()-1, **rfArgs)
    clf.fit(X, y)

    plt.matshow(clf.feature_importances_[ : width*height].reshape((width, height)))
    plt.show()


if __name__ == '__main__':

    width, height = 48, 48

    # write_train_data_to_files([(15, 15), (30, 30), (40, 40)])
    # write_test_data_to_files([(15, 15), (30, 30), (40, 40)])




    xdata = np.array(pandas.read_csv('/Users/jennyyuejin/K/NDSB/Data/X_train_48_48_simple.csv', header=None))
    ydata = np.array(pandas.read_csv('/Users/jennyyuejin/K/NDSB/Data/y.csv', header=None, dtype=int)).ravel()

    featureVals = xdata[:, width*height:]
    lastlayer = np.load('/Users/jennyyuejin/K/NDSB/Data/lastlayeroutput_train.npy')
    fullX = np.concatenate([lastlayer, featureVals], axis=1)


    # X_train, y = read_train_data(width, height)

    # x_fieldnames = np.array(['p_%i' % i for i in range(width*height)] + FEATURE_NAMES)
    # #
    # plot_feature_importances(fullX, ydata,
    #                          np.array(['p'+str(i) for i in range(lastlayer.shape[1])] + FEATURE_NAMES),
    #                          0.7, numEstimators=100, min_samples_split=15)

    # plot_pixel_importances(10, 20, lastlayer, ydata)

    # for minSampleSplit in [17]:
    #     print minSampleSplit
    #     print evaluate(fullX, ydata, RandomForestClassifier,
    #                    n_estimators=100, n_jobs=cpu_count()-1, min_samples_split=minSampleSplit)


    featureVals_test = np.array(pandas.read_csv('/Users/jennyyuejin/K/NDSB/Data/X_test_48_48_featureVals.csv', header=None))
    lastlayer_test = np.array(pandas.read_csv('/Users/jennyyuejin/K/NDSB/Data/lastlayerout_test.csv', header=None, sep=' '))
    fullX_test = np.concatenate([lastlayer_test, featureVals_test], axis=1)
    testFnames = list(np.array(pandas.read_table('/Users/jennyyuejin/K/NDSB/Data/testFnames.txt', header=None)).ravel())

    clf = RandomForestClassifier(n_estimators=100, min_samples_split=20)
    clf.fit(fullX, ydata)

    pred = np.zeros((fullX_test.shape[0], len(CLASS_NAMES)))
    pred[:, np.sort(list(set(ydata)))] = clf.predict_proba(fullX_test)

    make_submission_file(pred, testFnames, fNameSuffix='withLastLayer')