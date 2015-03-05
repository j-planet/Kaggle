__author__ = 'jennyyuejin'

import sys
sys.path.append('/Users/JennyYueJin/K/NDSB')

from global_vars import DATA_DIR, CLASS_MAPPING, CLASS_NAMES
# from Utilities.utilities import plot_feature_importances

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
from scipy import ndimage

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
                 'maxToMeanAreaRatio',
                 'rotationAngle'] \
                + ['nonEmptyPctl_%s' % str(p) for p in PERCENTILES] \
                + ['belowMeanPctl_%s' % str(p) for p in PERCENTILES]


def raw_moment(data, iord, jord):
    nrows, ncols = data.shape

    y, x = np.mgrid[:nrows, :ncols]
    data = data * x**iord * y**jord

    return data.sum()


def intertial_axis(_data):

    data = np.where(_data < 255, 1, 0)

    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_center = m10 / data_sum
    y_center = m01 / data_sum

    u11 = (raw_moment(data, 1, 1) - x_center * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_center * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_center * m01) / data_sum
    angle = 0.5 * np.arctan(2 * u11 / (u20 - u02))

    return x_center, y_center, angle


def trim_image(imData, bgValue=255):

    numRows, numCols = imData.shape

    tempCols = [np.all(imData[:, col] == np.repeat(bgValue, numRows)) for col in range(numCols)]
    tempRows = [np.all(imData[row, :] == np.repeat(bgValue, numCols)) for row in range(numRows)]

    if False not in tempRows or False not in tempCols:
        print 'The entire image is blank with background %i. Not trimming...' % bgValue
        return imData

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


def plot_rotation(imData):

    def plot_bars(x_bar, y_bar, angle):
        def plot_bar(r, x_center, y_center, angle, pattern):
            dx = r * np.cos(angle)
            dy = r * np.sin(angle)
            plt.plot([x_center - dx, x_center, x_center + dx],
                     [y_center - dy, y_center, y_center + dy], pattern)

        plot_bar(10, x_bar, y_bar, angle + np.radians(90), 'bo-')
        plot_bar(30, x_bar, y_bar, angle, 'ro-')

    def plot_subplot(pawprint):
        x_bar, y_bar, angle = intertial_axis(pawprint)
        plt.imshow(pawprint, cmap=cm.gray)
        plot_bars(x_bar, y_bar, angle)
        return angle

    plt.figure()
    angle = plot_subplot(imData)
    plt.title('Original')

    plt.figure()
    plot_subplot(ndimage.rotate(imData, np.degrees(angle)))
    plt.title('Rotated')


def largest_region(imData):

    belowMeanFilter = np.where(imData > np.mean(imData), 0., 1.0)
    dialated = morphology.dilation(belowMeanFilter, np.ones((3, 3)))
    regionLabels = (belowMeanFilter * measure.label(dialated)).astype(int)

    # calculate common region properties for each region within the segmentation
    regions = measure.regionprops(regionLabels)
    areas = [(None
              if sum(belowMeanFilter[regionLabels == region.label]) * 1.0 / region.area < 0.50
              else region.filled_area)
             for region in regions]

    if len(areas) > 0:

        regionMax = regions[np.argmax(areas)]

        # trim image to the max region
        regionMaxImg = trim_image(
            np.minimum(
                imData*np.where(regionLabels == regionMax.label, 1, 255),
                255))

        # rotate
        angle = intertial_axis(regionMaxImg)[2]
        rotatedRegionMaxImg = ndimage.rotate(regionMaxImg, np.degrees(angle))
        rotatedRegionMaxImg = trim_image(trim_image(rotatedRegionMaxImg, 0), 255)

    else:
        regionMax = None
        rotatedRegionMaxImg = None
        angle = 0

    return regionMax, rotatedRegionMaxImg, angle, regionLabels, regions, areas, belowMeanFilter, dialated


def create_features(imData, width, height, plot=False):
    """
    segment and return the minor-major axis ratio of the largest dark (above-average darkness) region
    :param imData:
    :param plot:
    :return:
    """

    totalArea = (np.where(imData < 255, 1, 0)).sum()

    regionMax, rotatedRegionMaxImg, angle, regionLabels, regions, areas, belowMeanFilter, dialated \
        = largest_region(imData)

    hasAreas = len(areas) > 0

    pixelData = resize(rotatedRegionMaxImg.astype('uint8') if hasAreas else imData,
                       (width, height)) \
        .reshape((1, width * height))

    # get a sense of elongatedness
    ratio = -1 if not hasAreas or regionMax.major_axis_length == 0 \
        else regionMax.minor_axis_length * 1.0 / regionMax.major_axis_length

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
        plt.imshow(dialated, cmap=cm.gray_r)
        sub3.set_title("Dilated Image")

        sub4 = plt.subplot(2, 2, 4)
        sub4.set_title("Labeled Image")
        plt.imshow(regionLabels)

        plt.figure()
        sns.kdeplot(belowMean, label='Below-Mean')
        sns.kdeplot(nonEmpty, label='Non-Empty')
        sns.kdeplot(imData.flatten(), label='Everything')

        if rotatedRegionMaxImg is not None:
            plt.figure()
            plt.imshow(rotatedRegionMaxImg, cmap=cm.gray)
            plt.title('Trimmed Largest Region')

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
                   max(areas) / np.mean(areas) if hasAreas else -1,
                   angle] \
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

import scipy

if __name__ == '__main__':
    img = 1 - resize(trim_image(imread('/Users/jennyyuejin/K/NDSB/Data/train/chaetognath_sagitta/72705.jpg')), (48, 48))

    degrees = [0, 10, 45, 90, 135, 180, 235, 270, -10]
    plt.figure()

    for i, degree in enumerate(degrees):
        print np.ceil(len(degrees)/3.), i+1
        plt.subplot(3, np.ceil(len(degrees)/3.), i+1, title=str(degree))


        plt.imshow(resize(trim_image(ndimage.rotate(img, degree)), img.shape), cmap=cm.gray_r)


    plt.show()