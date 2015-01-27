__author__ = 'jennyyuejin'

import sys
sys.path.append('/Users/JennyYueJin/K/NDSB')

from global_vars import DATA_DIR, CLASS_MAPPING, CLASS_NAMES

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
plt.ioff()


def trim_image(imData, bgValue=255):

    numRows, numCols = imData.shape

    tempCols = [np.all(imData[:, col] == np.repeat(bgValue, numRows)) for col in range(numCols)]
    tempRows = [np.all(imData[row, :] == np.repeat(bgValue, numCols)) for row in range(numRows)]

    firstCol = tempCols.index(False)
    firstRow = tempRows.index(False)
    lastCol = -(1 + tempCols[::-1].index(False))
    lastRow = -(1 + tempRows[::-1].index(False))

    if lastRow == -1:
        if lastCol == -1:
            return imData[firstRow:, firstCol:]
        else:
            return imData[firstRow:, firstCol:(lastCol+1)]
    else:
        if lastCol == -1:
            return imData[firstRow:(lastRow+1), firstCol:]
        else:
            return imData[firstRow:(lastRow+1), firstCol:(lastCol+1)]


def num_lines_in_file(fpath):
    return int(subprocess.check_output('wc -l %s' % fpath, shell=True).strip().split()[0])


def list_dirs(path):
    return [t[0] for t in os.walk(path)]


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

PERCENTILES = [0.1, 1, 5, 10, 20, 30, 50, 75]


# names:
names = ['axisRatio',                  # minor to major axis ratio,
         'fracBelowMean',         # fraction of below-mean pixels
         'fracNonEmpty',  # frations of non-empty (i.e. < 255) pixels
         'meanNonEmpty',           # mean non-empty pixel value
         'numRegions',               # number of regions
         'totalArea',
         'maxArea',                 # area covered by the largest region
         'fracMaxAreaToTotal',
         'avgArea',             # average area covered
         'maxToMinAreaRatio',    # largest to small region area ratio
         'maxToMeanAreaRatio'] \
        + ['nonEmptyPctl_%s' % str(p) for p in PERCENTILES] \
        + ['belowMeanPctl_%s' % str(p) for p in PERCENTILES]


def get_minor_major_ratio(imData, width, height, plot=False):
    """
    segment and return the minor-major axis ratio of the largest dark (above-average darkness) region
    :param imData:
    :param plot:
    :return:
    """

    totalArea = (np.where(imData < 255, 1, 0)).sum()
    belowMeanFilter = np.where(imData > np.mean(imData), 0., 1.0)
    imdilated = morphology.dilation(belowMeanFilter, np.ones((3, 3)))
    labels = (belowMeanFilter * measure.label(imdilated)).astype(int)

    # calculate common region properties for each region within the segmentation
    regions = measure.regionprops(labels)

    areas = [(None
              if sum(belowMeanFilter[labels == region.label])*1.0/region.area < 0.50
              else region.filled_area)
             for region in regions]
    hasAreas = len(areas) > 0

    pixelData = None

    if hasAreas:
        regionmax = regions[np.argmax(areas)]

        # trim image to the max region
        mask = np.where(labels == regionmax.label, 1, 255)
        im_lr = np.minimum(imData*mask, 255)    # largest-region image
        trimmedMax = trim_image(im_lr)
        pixelData = resize(trimmedMax.astype('uint8'), (width, height))

        if plot:
            plt.figure()
            plt.imshow(im_lr, cmap=cm.gray)
            plt.title('Largest Region')

            plt.figure()
            plt.imshow(trimmedMax, cmap=cm.gray)
            plt.title('Trimmed Largest Region')
    else:
        pixelData = resize(imData, (width, height))

    pixelData = pixelData.reshape((width * height, ))

    # get a sense of elongatedness
    ratio = -1 if not hasAreas or regionmax.major_axis_length == 0 \
        else regionmax.minor_axis_length * 1.0 / regionmax.major_axis_length

    nonEmpty = imData[imData < 255]
    belowMean = imData[imData < imData.mean()]

    if plot:
        plt.figure(figsize=(8, 8))
        sub1 = plt.subplot(2, 2, 1)
        plt.imshow(imData, cmap=cm.gray)
        sub1.set_title("Original Image")

        sub2 = plt.subplot(2, 2, 2)
        plt.imshow(belowMeanFilter, cmap=cm.gray_r)
        sub2.set_title("Thresholded Image")

        sub3 = plt.subplot(2, 2, 3)
        plt.imshow(imdilated, cmap=cm.gray_r)
        sub3.set_title("Dilated Image")

        sub4 = plt.subplot(2, 2, 4)
        sub4.set_title("Labeled Image")
        plt.imshow(labels)

        plt.figure(figsize=(5, 5))
        plt.imshow(np.where(labels == regionmax.label, 1.0, 0.0))

        plt.figure()
        sns.kdeplot(belowMean, label='Below-Mean')
        sns.kdeplot(nonEmpty, label='Non-Empty')
        sns.kdeplot(imData.flatten(), label='Everything')

        plt.show(block=True)

    featureData = [ratio,                  # minor to major axis ratio,
                   belowMeanFilter.sum()*1./imData.size,         # fraction of below-mean pixels
                   nonEmpty.size * 1. / imData.size,  # frations of non-empty (i.e. < 255) pixels
                   nonEmpty.mean(),           # mean non-empty pixel value
                   len(regions),               # number of regions
                   totalArea,
                   max(areas) if hasAreas else -1,                 # area covered by the largest region
                   max(areas)*1./totalArea if hasAreas else -1,
                   np.mean(areas) if hasAreas else -1,             # average area covered
                   max(areas) / min(areas) if hasAreas else -1,    # largest to small region area ratio
                   max(areas) / np.mean(areas) if hasAreas else -1] \
                  + stats.scoreatpercentile(nonEmpty, PERCENTILES) \
                  + stats.scoreatpercentile(belowMean, PERCENTILES)    # percentiles of below-mean pixel values

    return np.append(pixelData, featureData)


def create_test_data_table(testFListFpath, width, height):

    numSamples = num_lines_in_file(testFListFpath)
    X = np.zeros((numSamples, width * height + len(names)))
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


def append_features(X, imgWidth, imgHeight, featureFuncsNnum):
    """
    :param X: numpy array of size imgWidth x imgHeight
    :param imgWidth:
    :param imgHeight:
    :param featureFuncsNnum: [function (returns a list of values), number of features]
    :return:
    """

    featureMat = np.zeros((X.shape[0], sum(num for _, num in featureFuncsNnum)))

    for rowInd in range(X.shape[0]):

        img = X[rowInd, :].reshape((imgWidth, imgHeight))
        featureMat[rowInd, :] = np.array(list(itertools.chain([func(img) for func, _ in featureFuncsNnum])))

    return np.concatenate([X, featureMat], axis=1)


def create_training_data_table(trainFListFpath, width, height):

    """
    :param trainDatadir:
    :param width:
    :param height:
    :return:    X (numpy array of size numImgs x (maxWidth x maxHeight),
                y (numpy array of size (numImgs,))
    """

    numSamples = num_lines_in_file(trainFListFpath)
    X = np.zeros((numSamples, width * height + len(names)))
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


def plot_ratio_distns_for_pairs(minimumSize=20):
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
                 label=namesClasses[classind1], normed=1)

        plt.hist(df[df["class"] == classind2]["ratio"].values,
                 alpha=0.5, bins=bins, label=namesClasses[classind2], normed=1)

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


def write_data_to_files(sizes, trainFListFpath, testFListFpath, outputDir = DATA_DIR):
    """
    most likely to be called once only
    :param sizes: list of sizes. [(width_0, height_0), ..., (width_n, height_n)]
    :return: file names
    """

    for width, height in sizes:

        X_train, y = create_training_data_table(trainFListFpath, width, height)
        # train X
        np.savetxt(os.path.join(outputDir, 'X_train_%i_%i.csv' % (width, height)),
                   X_train, delimiter=',')

        # train y
        np.savetxt(os.path.join(outputDir, 'y.csv'), y, delimiter=',')


        X_test, testFnames = create_test_data_table(testFListFpath, width, height)
        # test X
        pandas.DataFrame(X_test, index=testFnames). \
            to_csv(os.path.join(outputDir, 'X_test_%i_%i.csv' % (width, height)), header=False)


if __name__ == '__main__':


    width, height = 25, 25
    write_data_to_files([(25, 25), (35, 35)],
                        os.path.join(DATA_DIR, 'trainFnames.txt'),
                        os.path.join(DATA_DIR, 'testFnames.txt'))

    # X_train, y = create_training_data_table('/Users/jennyyuejin/K/NDSB/Data/trainFnames.txt', 25, 25)
    # X_test, testFnames = create_test_data_table('/Users/jennyyuejin/K/NDSB/Data/testFnames.txt', 25, 25)
    #
    # # print "CV-ing"
    # scores = cross_validation.cross_val_score(RandomForestClassifier(n_estimators=100, n_jobs=cpu_count()-1),
    #                                           X_train, y, cv=5, n_jobs=cpu_count()-1)
    #
    # print "Accuracy of all classes:", np.mean(scores)
    #
    # # Get the probability predictions for computing the log-loss function
    # # prediction probabilities number of samples, by number of classes
    # y_pred = y * 0
    # y_pred_mat = np.zeros((len(y), len(CLASS_NAMES)))   # forcing all class names, for testing with partial data
    #
    # for trainInd, testInd in KFold(y, n_folds=5):
    #     clf = RandomForestClassifier(n_estimators=100, n_jobs=cpu_count()-1)
    #     clf.fit(X_train[trainInd, :], y[trainInd])
    #
    #     y_pred[testInd] = clf.predict(X_train[testInd, :])
    #     y_pred_mat[testInd, :][:, np.sort(list(set(y)))] = clf.predict_proba(X_train[testInd, :])
    #
    # print '>>>>>> Classification Report'
    # print classification_report(y, y_pred, target_names=CLASS_NAMES)
    #
    # print '\n>>>>>>>Multi-class Log Loss =', multiclass_log_loss(y, y_pred_mat)
    #
    # # make predictions and write to file
    # clf = RandomForestClassifier(n_estimators=100, n_jobs=cpu_count()-1)
    # clf.fit(X_train, y)
    # y_test_pred = np.zeros((X_test.shape[0], len(CLASS_NAMES)))
    # y_test_pred[:, np.sort(list(set(y)))] = clf.predict_proba(X_test)
    # pandas.DataFrame(y_test_pred, index=testFnames).reset_index() \
    #     .to_csv(os.path.join(DATA_DIR, 'submissions', 'base_%s.csv' % datetime.date.today().strftime('%b%d%Y')),
    #             header = ['image'] + CLASS_NAMES, index=False)
#

