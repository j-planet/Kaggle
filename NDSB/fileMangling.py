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

from features import get_minor_major_ratio, FEATURE_NAMES


plt.ioff()

def num_lines_in_file(fpath):
    return int(subprocess.check_output('wc -l %s' % fpath, shell=True).strip().split()[0])


def list_dirs(path):
    return [t[0] for t in os.walk(path)]


def create_test_data_table(testFListFpath, width, height):

    numSamples = num_lines_in_file(testFListFpath)
    X = np.zeros((numSamples, width * height + len(FEATURE_NAMES)))
    imgNames = [None]*numSamples

    printIs = [int(i/100.*numSamples) for i in np.arange(100)]
    i = 0

    for fpath in open(testFListFpath):
        fpath = fpath.strip()

        imData = imread(os.path.join(DATA_DIR, fpath))
        X[i, :] = get_minor_major_ratio(imData, width, height)

        imgNames[i] = fpath.split(os.sep)[-1]

        i += 1

        if i in printIs:
            print '%i%% done...' % (100. * i/numSamples)

    return X, imgNames


def create_training_data_table(trainFListFpath, width, height):

    """
    :param trainDatadir:
    :param width:
    :param height:
    :return:    X (numpy array of size numImgs x (maxWidth x maxHeight),
                y (numpy array of size (numImgs,))
    """

    numSamples = num_lines_in_file(trainFListFpath)
    X = np.zeros((numSamples, width * height + len(FEATURE_NAMES)))
    y = np.zeros((numSamples, ))

    printIs = [int(i/100.*numSamples) for i in np.arange(100)]
    i = 0

    for fpath in open(trainFListFpath):
        fpath = fpath.strip()

        className = fpath.split(os.sep)[-2]
        classLabel = CLASS_MAPPING[className]

        imData = imread(os.path.join(DATA_DIR, fpath))
        X[i, :] = get_minor_major_ratio(imData, width, height)
        y[i] = classLabel

        i += 1

        if i in printIs:
            print '%i%% done...' % (100. * i/numSamples)

    return X, y.astype(int)


def write_data_to_files(sizes,
                        trainFListFpath = os.path.join(DATA_DIR, 'trainFnames.txt'),
                        testFListFpath = os.path.join(DATA_DIR, 'testFnames.txt'),
                        outputDir = DATA_DIR):
    """
    most likely to be called once only
    :param sizes: list of sizes. [(width_0, height_0), ..., (width_n, height_n)]
    :return: file names
    """

    for width, height in sizes:

        X_train, y = create_training_data_table(trainFListFpath, width, height)
        # train X
        np.savetxt(os.path.join(outputDir, 'X_train_%i_%i_rf.csv' % (width, height)),
                   X_train, delimiter=',')

        # train y
        np.savetxt(os.path.join(outputDir, 'y.csv'), y, delimiter=',')


        X_test, testFnames = create_test_data_table(testFListFpath, width, height)
        # test X
        pandas.DataFrame(X_test, index=testFnames). \
            to_csv(os.path.join(outputDir, 'X_test_%i_%i_rf.csv' % (width, height)), header=False)


def read_train_data(width, height, inputDir = DATA_DIR, isTiny = False):
    """
    :param width:
    :param height:
    :param inputDir:
    :param isTiny:
    :return: x train, y train, x test, xtest filenames
    """

    print '======= Reading Data ======='

    # ----- read x -----
    X_train = np.array(pandas.read_csv(os.path.join(
        inputDir, '%sX_train_%i_%i.csv' % ('tiny' if isTiny else '', width, height)),
                                       header=None))
    # ----- read y -----
    y = np.array(pandas.read_csv(os.path.join(inputDir, '%sy.csv' % ('tiny' if isTiny else '')),
                                 header=None)).flatten()

    print 'DONE reading data. :)'

    return X_train, y.astype(int)


def read_test_data(width, height, inputDir = DATA_DIR, isTiny = False):
    """
    :param width:
    :param height:
    :param inputDir:
    :param isTiny:
    :return: x train, y train, x test, xtest filenames
    """

    # ----- read x -----
    return read_test_data_given_path(os.path.join(inputDir, '%sX_test_%i_%i.csv' % ('tiny' if isTiny else '', width, height)))


def read_test_data_given_path(fpath):
    print '======= Reading Data ======='

    # ----- read x -----
    X_test = np.array(pandas.read_csv(fpath, header=None))

    print 'DONE reading data. :)'

    return X_test[:, 1:], X_test[:, 0]


if __name__ == '__main__':

    width, height = 25, 25
    write_data_to_files([(25, 25)])