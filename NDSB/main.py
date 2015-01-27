__author__ = 'jennyyuejin'

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
from scipy import ndimage
from skimage.feature import peak_local_max

from global_vars import DATA_DIR, CLASS_MAPPING, CLASS_NAMES


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


def get_minor_major_ratio(_im, plot=False):
    """
    segment and return the minor-major axis ratio of the largest dark (above-average darkness) region
    :param _im:
    :param plot:
    :return:
    """

    # First we threshold the image by only taking values below (i.e. darker than) the mean to reduce noise in the image
    # to use later as a mask
    try:
        imthr = np.where(_im > np.mean(_im), 0., 1.0)
        imdilated = morphology.dilation(imthr, np.ones((4, 4)))
        labels = (imthr * measure.label(imdilated)).astype(int)

        # calculate common region properties for each region within the segmentation
        regionmax = get_largest_region(props=measure.regionprops(labels), labelmap=labels, imagethres=imthr)

        # get a sense of elongatedness
        ratio = -1 if regionmax is None or regionmax.major_axis_length == 0 \
            else regionmax.minor_axis_length * 1.0 / regionmax.major_axis_length

        if plot:
            plt.figure(figsize=(8, 8))
            sub1 = plt.subplot(2, 2, 1)
            plt.imshow(_im, cmap=cm.gray)
            sub1.set_title("Original Image")

            sub2 = plt.subplot(2, 2, 2)
            plt.imshow(imthr, cmap=cm.gray_r)
            sub2.set_title("Thresholded Image")

            sub3 = plt.subplot(2, 2, 3)
            plt.imshow(imdilated, cmap=cm.gray_r)
            sub3.set_title("Dilated Image")

            sub4 = plt.subplot(2, 2, 4)
            sub4.set_title("Labeled Image")
            plt.imshow(labels)

            plt.figure(figsize=(5, 5))
            plt.imshow(np.where(labels == regionmax.label, 1.0, 0.0))

            plt.show()
    except:
        return [-1]

    return [ratio]


def create_test_data_table(testDataDir, maxWidth, maxHeight):

    imageSize = maxWidth * maxHeight
    num_features = imageSize
    imgNames = [name for name in os.walk(testDataDir).next()[2] if name[-4:].lower()=='.jpg']

    report = [int((j+1)*len(imgNames)/100.) for j in range(100)]
    X = np.zeros((len(imgNames), num_features), dtype=float)

    for i, imgFname in enumerate(imgNames):

        image = imread("{0}{1}{2}".format(testDataDir, os.sep, imgFname), as_grey=True)

        X[i, :] = resize(image, (maxWidth, maxHeight)).reshape((imageSize, ))

        # report progress for each 5% done
        if i in report:
            print np.ceil(i *100.0 / len(imgNames)), "% done"

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


def create_training_data_table(trainDatadir, maxWidth, maxHeight):

    """
    :param trainDatadir:
    :param maxWidth:
    :param maxHeight:
    :return:    X (numpy array of size numImgs x (maxWidth x maxHeight),
                y (numpy array of size (numImgs,))
    """

    directory_names = list(set(list_dirs(trainDatadir)).difference([trainDatadir]))

    #get the total training images
    numberofImages = 0

    for classFolder in directory_names:
        for imgFname in os.walk(classFolder).next()[2]:
            # Only read in the images
            if imgFname[-4:] != ".jpg":
                continue
            numberofImages += 1


    imageSize = maxWidth * maxHeight
    num_rows = numberofImages # one row for each image in the training dataset
    num_cols = imageSize

    # X is the feature vector with one row of features per image
    # consisting of the pixel values and our metric
    X = np.zeros((num_rows, num_cols), dtype=float)
    y = np.zeros(num_rows)      # y is the numeric class label

    i = 0
    report = [int((j+1)*num_rows/100.) for j in range(100)]

    print "Reading images"

    # Navigate through the list of directories
    for classFolder in directory_names:

        # Append the string class name for each class
        curClass = classFolder.split(os.sep)[-1].strip()
        curLabel = CLASS_MAPPING[curClass]

        print 'Class:', curClass, curLabel

        for imgFname in os.walk(classFolder).next()[2]:

            if imgFname[-4:] != ".jpg":
                continue

            # Read in the images and create the features
            nameFileImage = "{0}{1}{2}".format(classFolder, os.sep, imgFname)
            image = imread(nameFileImage, as_grey=True)

            # print 'Resizing from %ix%i to %ix%i' % (image.shape[0], image.shape[1], maxPixel, maxPixel)
            # Store the rescaled image pixels and the axis ratio
            X[i, :] = np.reshape(resize(image, (maxWidth, maxHeight)), (1, imageSize))

            # Store the classlabel
            y[i] = curLabel


            # report progress for each 5% done
            if i in report:
                print np.ceil(i * 100.0 / num_rows), "% done"

            i += 1


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


def write_data_to_files(sizes, outputDir = DATA_DIR):
    """
    most likely to be called once only
    :param sizes: list of sizes. [(width_0, height_0), ..., (width_n, height_n)]
    :return: file names
    """

    for width, height in sizes:

        X_train, y = create_training_data_table(os.path.join(DATA_DIR, 'train'), width, height)
        X_test, testFnames = create_test_data_table(os.path.join(DATA_DIR, 'test'), width, height)

        # train X
        np.savetxt(os.path.join(outputDir, 'X_train_%i_%i.csv' % (width, height)),
                   X_train, delimiter=',')

        # train y
        np.savetxt(os.path.join(outputDir, 'y.csv'), y, delimiter=',')

        # test X
        pandas.DataFrame(X_test, index=testFnames). \
            to_csv(os.path.join(outputDir, 'X_test_%i_%i.csv' % (width, height)), header=False)


def read_data(width, height, featureFuncsNnum, inputDir = DATA_DIR, isTiny = False):
    """
    :param width:
    :param height:
    :param featureFuncsNnum:
    :param inputDir:
    :param isTiny:
    :return: x train, y train, x test, xtest filenames
    """

    print '======= Reading Data ======='

    # ----- read x -----
    X_train_woFeatures = np.array(pandas.read_csv(os.path.join(
        inputDir, '%sX_train_%i_%i.csv' % ('tiny' if isTiny else '', width, height)),
                                                  header=None))
    X_train = append_features(X_train_woFeatures, width, height, featureFuncsNnum)

    X_test_woFeatures = np.array(pandas.read_csv(
        os.path.join(inputDir, '%sX_test_%i_%i.csv' % ('tiny' if isTiny else '', width, height)),
        header=None))
    X_test = append_features(X_test_woFeatures[:, 1:], width, height, featureFuncsNnum)

    # ----- read y -----
    y = np.array(pandas.read_csv(os.path.join(inputDir, '%sy.csv' % ('tiny' if isTiny else '')),
                                 header=None)).flatten()

    print 'DONE reading data. :)'

    return X_train, y.astype(int), X_test, X_test_woFeatures[:, 0]

if __name__ == '__main__':
    width, height = 25, 25
    X_train, y, X_test, testFnames = read_data(width, height, [(get_minor_major_ratio, 1)], isTiny=False)

    # print "CV-ing"
    # scores = cross_validation.cross_val_score(RandomForestClassifier(n_estimators=100, n_jobs=cpu_count()-1),
    #                                           X_train, y, cv=5, n_jobs=cpu_count()-1)
    #
    # print "Accuracy of all classes:", np.mean(scores)

    # Get the probability predictions for computing the log-loss function
    # prediction probabilities number of samples, by number of classes
    y_pred = y * 0
    y_pred_mat = np.zeros((len(y), len(CLASS_NAMES)))   # forcing all class names, for testing with partial data

    for trainInd, testInd in KFold(y, n_folds=5):
        clf = RandomForestClassifier(n_estimators=100, n_jobs=cpu_count()-1)
        clf.fit(X_train[trainInd, :], y[trainInd])

        y_pred[testInd] = clf.predict(X_train[testInd, :])
        y_pred_mat[testInd, :][:, np.sort(list(set(y)))] = clf.predict_proba(X_train[testInd, :])

    print '>>>>>> Classification Report'
    print classification_report(y, y_pred, target_names=CLASS_NAMES)

    print '\n>>>>>>>Multi-class Log Loss =', multiclass_log_loss(y, y_pred_mat)

    # make predictions and write to file
    clf = RandomForestClassifier(n_estimators=100, n_jobs=cpu_count()-1)
    clf.fit(X_train, y)
    y_test_pred = np.zeros((X_test.shape[0], len(CLASS_NAMES)))
    y_test_pred[:, np.sort(list(set(y)))] = clf.predict_proba(X_test)
    pandas.DataFrame(y_test_pred, index=testFnames).reset_index() \
        .to_csv(os.path.join(DATA_DIR, 'submissions', 'base_%s.csv' % datetime.date.today().strftime('%b%d%Y')),
                header = ['image'] + CLASS_NAMES, index=False)
