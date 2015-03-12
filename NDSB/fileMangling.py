__author__ = 'jennyyuejin'

import sys
sys.path.append('/Users/JennyYueJin/K/NDSB')

from global_vars import DATA_DIR, DATA_DIR_TEST, CLASS_MAPPING, CLASS_NAMES
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
from sklearn.cross_validation import StratifiedKFold as KFold, StratifiedShuffleSplit
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
import theano
import scipy.stats as stats
import seaborn as sns
from skimage.feature import peak_local_max

from features import create_features, FEATURE_NAMES, trim_image
from scipy.ndimage import rotate

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
        try:
            fpath = fpath.strip()

            imData = imread(os.path.join(DATA_DIR_TEST, fpath))
            X[i, :] = create_features(imData, width, height)

            imgNames[i] = fpath.split(os.sep)[-1]

        except Exception as e:

            print 'Skipping image %s due to error %s' % (fpath, e.message)

        i += 1

        if i in printIs:
            print '%i%% done...' % (100. * i/numSamples)

    return X, imgNames


def write_test_data_table_simple(testFListFpath, width, height, xFpath, angles=[], fmt='%.5e'):

    numSamples = num_lines_in_file(testFListFpath)
    imgSize = width * height

    xFile = file(xFpath, 'w')
    xFile = file(xFpath, 'a')

    printIs = [int(i/100.*numSamples) for i in np.arange(100)]


    for i, fname in enumerate(open(testFListFpath)):

        try:
            fname = fname.strip()
            # print imread(os.path.join(DATA_DIR_TEST, fname))
            img = 1 - trim_image(imread(os.path.join(DATA_DIR_TEST, fname)))

            # write original image
            origImg = resize(img, (width, height)).ravel().reshape(1, imgSize).astype(theano.config.floatX)
            np.savetxt(xFile, origImg, fmt=fmt, delimiter=',')

            # write angled images
            for angle in angles:
                angledImg = resize(rotate(img, angle), (width, height)).ravel().reshape(1, imgSize).astype(theano.config.floatX)
                np.savetxt(xFile, angledImg, fmt=fmt, delimiter=',')

        except Exception as e:

            print 'Skipping image %s due to error %s' % (fname, e.message)

        if i in printIs:
            print '%i%% done...' % (100. * i/numSamples)


    xFile.close()

def create_test_data_table_simple(testFListFpath, width, height):

    numSamples = num_lines_in_file(testFListFpath)
    X = np.zeros((numSamples, width * height))
    imgNames = [None]*numSamples

    printIs = [int(i/100.*numSamples) for i in np.arange(100)]
    i = 0

    for fpath in open(testFListFpath):
        try:
            fpath = fpath.strip()

            imData = imread(os.path.join(DATA_DIR, fpath))
            X[i, :] = resize(trim_image(imData), (width, height)).ravel()

            imgNames[i] = fpath.split(os.sep)[-1]

        except Exception as e:
            print 'Skipping image %s due to error %s' % (fpath, e.message)

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
        try:
            fpath = fpath.strip()

            className = fpath.split(os.sep)[-2]
            classLabel = CLASS_MAPPING[className]

            imData = imread(os.path.join(DATA_DIR, fpath))
            X[i, :] = create_features(imData, width, height)
            y[i] = classLabel
        except Exception as e:
            print 'Skipping image %s due to error %s' % (fpath, e.message)

        i += 1

        if i in printIs:
            print '%i%% done...' % (100. * i/numSamples)

    return X, y.astype(int)


def make_append_file(fpath):
    """
    creates and returns file object in append mode
    :param fpath:
    :return:
    """

    if fpath is None:
        return None

    file(fpath, 'w')
    return file(fpath, 'a')


def create_images(dataFpath, width, height, angles=[]):
    """
    returns the resized image and its rotated versions
    :param dataFpath:
    :param width:
    :param height:
    :param angles:
    :return: list of straight img, rotated images
    """

    imgSize = width * height


    def _resize_n_cast(_img):
        return resize(_img, (width, height)).ravel().reshape(1, imgSize).astype(theano.config.floatX)


    origImg = 1 - trim_image(imread(dataFpath, as_grey=True), bgValue=255)

    straightImg = _resize_n_cast(origImg)
    rotatedImgs = [_resize_n_cast(rotate(origImg, angle)) for angle in angles]

    return [straightImg] + rotatedImgs



def write_training_data_table_angles(trainFListFpath,
                                     width, height, angles=[], fmt='%.7e',
                                     xFpath_train=None, yFpath_train=None,
                                     xFpath_val=None, yFpath_val=None,
                                     validation_size = 0.1,
                                     sampleXFpath_train=None, sampleYFpath_train=None, sampleXFpath_val=None, sampleYFpath_val=None,
                                     sampleFrequency=None):

    """
    :param trainDatadir:
    :param width:
    :param height:
    :return:    X (numpy array of size numImgs x (maxWidth x maxHeight),
                y (numpy array of size (numImgs,))
    """

    def _img_to_file(_imgData, _file):
        np.savetxt(_file, _imgData, fmt=fmt, delimiter=',')


    def _imgs_to_file(imgs, _file):
        if _file is not None:
            for curImg in imgs:
                _img_to_file(curImg, _file)


    def _y_to_file(_y, _file):
        if _file is not None:
            np.savetxt(_file, np.ones((1 + len(angles))) * _y, fmt=fmt, delimiter=',')


    def _close(_file):
        if _file is not None:
            _file.close()

    # ---- initialize files ----
    # files are automatically set to None if paths are not given
    xFile_train = make_append_file(xFpath_train)
    yFile_train = make_append_file(yFpath_train)

    xFile_val = make_append_file(xFpath_val)
    yFile_val = make_append_file(yFpath_val)

    sampleXFile_train = make_append_file(sampleXFpath_train)
    sampleYFile_train = make_append_file(sampleYFpath_train)

    sampleXFile_val = make_append_file(sampleXFpath_val)
    sampleYFile_val = make_append_file(sampleYFpath_val)


    # ---- read all Ys ----
    classLabels, fpaths = zip(*[(CLASS_MAPPING[fpath.strip().split(os.sep)[-2]], fpath.strip()) for fpath in open(trainFListFpath)])
    numSamples = len(classLabels)

    # ---- split into training and validation datasets ----
    trainInds, valInds = StratifiedShuffleSplit(classLabels, n_iter=1, test_size=validation_size)._iter_indices().next()


    # ---- output to files ----
    printIs = [int(i/100.*numSamples) for i in np.arange(100)]

    for i, curY in enumerate(classLabels):

        curDatapath = fpaths[i]

        if i in printIs: print '%i%% done...' % (100. * i/numSamples)
        assert i in trainInds or i in valInds, '%i not found in either train indices or validation indices.' % i

        try:

            curImgs = create_images(ddf(curDatapath), width, height, angles=angles)

            _imgs_to_file(curImgs, xFile_train if i in trainInds else xFile_val)
            _y_to_file(curY, yFile_train if i in trainInds else yFile_val)

            # sample files
            if sampleFrequency is not None and i % sampleFrequency==0:
                _imgs_to_file(curImgs, sampleXFile_train if i in trainInds else sampleXFile_val)
                _y_to_file(curY, sampleYFile_train if i in trainInds else sampleYFile_val)

        except Exception as e:
            print 'Skipping image %s due to error: %s' % (curDatapath, e.message)


    # close files
    for f in xFile_train, xFile_val, yFile_train, yFile_val, sampleXFile_train, sampleXFile_val, sampleYFile_train, sampleYFile_val:
        _close(f)


def write_train_data_to_files(sizes,
                              trainFListFpath = os.path.join(DATA_DIR, 'trainFnames.txt'),
                              outputDir = DATA_DIR):
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


def write_test_data_to_files(sizes,
                             testFListFpath = os.path.join(DATA_DIR, 'testFnames.txt'),
                             outputDir = DATA_DIR):

    for width, height in sizes:
        X_test, testFnames = create_test_data_table(testFListFpath, width, height)
        # test X
        pandas.DataFrame(X_test, index=testFnames). \
            to_csv(os.path.join(outputDir, 'X_test_%i_%i.csv' % (width, height)), header=False)


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


def make_submission_file(pred, testFnames,
                         outputDir = os.path.join(DATA_DIR, 'submissions'), fNameSuffix=''):

    res = pandas.DataFrame(pred, index=testFnames).reset_index()
    res.columns = ['image'] + CLASS_NAMES

    outputFpath = os.path.join(outputDir,
                               '%s%s.csv' %
                               (datetime.date.today().strftime('%b%d%Y'),
                                '_' + fNameSuffix if len(fNameSuffix) > 0 else ''))

    print 'Writing predictions to', outputFpath
    res.to_csv(outputFpath, index=False)

    return outputFpath


def str_angles(angles):
    """  doesn't output negative signs
    :param angles:
    :return:
    """
    return ''.join(str(abs(a)) for a in angles)

def str_shape(width, height):
    return '%i_%i' % (width, height)

def ddf(*fpaths):
    return os.path.join(DATA_DIR, *fpaths)


if __name__ == '__main__':

    width, height = 48, 48
    angles = [-1, 1, -2, 2]

    # write_train_data_to_files([(25, 25)])

    # x_train, _ = create_training_data_table(os.path.join(DATA_DIR, 'trainFnames.txt'), width, height)
    # np.savetxt(os.path.join(DATA_DIR, 'X_train_%i_%i.csv' % (width, height)), x_train, delimiter=',')
    #
    # x_test, testFnames = create_test_data_table(os.path.join(DATA_DIR, 'testFnames.txt'), width, height)
    # pandas.DataFrame(x_test, index=testFnames). \
    #     to_csv(os.path.join(DATA_DIR, 'X_test_%i_%i.csv' % (width, height)), header=False)

    # x_train, _ = create_training_data_table_simple(os.path.join(DATA_DIR, 'trainFnames.txt'), width, height)
    # np.savetxt(os.path.join(DATA_DIR, 'X_train_%i_%i_simple.csv' % (width, height)), x_train, delimiter=',')
    #
    # x_test, testFnames = create_test_data_table_simple(os.path.join(DATA_DIR, 'testFnames.txt'), width, height)
    # pandas.DataFrame(x_test, index=testFnames). \
    #     to_csv(os.path.join(DATA_DIR, 'X_test_%i_%i_simple.csv' % (width, height)), header=False)
    #

    # write_test_data_table_simple(os.path.join(DATA_DIR, 'testFnames.txt'),
    #                              width, height,
    #                              xFpath=os.path.join(DATA_DIR, 'X_test_48_48_-112233.csv'),
    #                              angles=[-1, 1, -2, 2, -3, 3],
    #                              fmt='%.4e'
    # )

    baseDir = lambda *args: os.path.join(ddf(str_shape(width, height), str_angles(angles=angles)), *args)

    write_training_data_table_angles(ddf('trainFnames.txt'),
                                     width, height, angles=angles, fmt='%.4e', validation_size=0.1,

                                     xFpath_train=baseDir('train', 'fullX.csv'),
                                     xFpath_val=baseDir('val', 'fullX.csv'),
                                     yFpath_train=baseDir('train', 'fullY.csv'),
                                     yFpath_val=baseDir('val', 'fullY.csv'),

                                     sampleFrequency=10,
                                     sampleXFpath_train=baseDir('train', 'sampleX.csv'),
                                     sampleXFpath_val=baseDir('val', 'sampleX.csv'),
                                     sampleYFpath_train=baseDir('train', 'sampleY.csv'),
                                     sampleYFpath_val=baseDir('val', 'sampleY.csv'),
                                     )

