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

plt.ioff()


PERCENTILES = [0.1, 1, 5, 10, 20, 30, 50, 75]

# names:
FEATURE_NAMES = ['axisRatio',                  # minor to major axis ratio,
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
        pixelData = resize(imData, (width, height))
        # pixelData = resize(trimmedMax.astype('uint8'), (width, height))

        if plot:
            plt.figure()
            plt.imshow(im_lr, cmap=cm.gray)
            plt.title('Largest Region')

            plt.figure()
            plt.imshow(trimmedMax, cmap=cm.gray)
            plt.title('Trimmed Largest Region')
    else:
        pixelData = resize(imData, (width, height))

    pixelData = pixelData.reshape((1, width * height))

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
