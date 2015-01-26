__author__ = 'jennyyuejin'

#Import libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
import glob
import os
from pprint import pprint
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
# make graphics inline
# %matplotlib inline

DATA_DIR = '/Users/jennyyuejin/K/NDSB/Data'


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
        return -1

    return ratio


def read_test_data(testDataDir, maxWidth, maxHeight, featureFuncsNnum=[]):

    imageSize = maxWidth * maxHeight
    num_features = imageSize + sum([t[1] for t in featureFuncsNnum])
    imgNames = [name for name in os.walk(testDataDir).next()[2] if name[-4:].lower()=='.jpg']

    report = [int((j+1)*len(imgNames)/100.) for j in range(100)]
    X = np.zeros((len(imgNames), num_features), dtype=float)

    for i, imgFname in enumerate(imgNames):

        image = imread("{0}{1}{2}".format(testDataDir, os.sep, imgFname), as_grey=True)
        image = resize(image, (maxWidth, maxHeight))

        # Store the rescaled image pixels and the axis ratio
        curRow = np.reshape(image, (imageSize, ))

        #  add additional features
        for func, _ in featureFuncsNnum:
            curRow = np.append(curRow, func(image))

        X[i, :] = curRow

        # report progress for each 5% done
        if i in report:
            print np.ceil(i *100.0 / len(imgNames)), "% done"

    return X, imgNames


def read_training_data(trainDatadir, maxWidth, maxHeight, featureFuncsNnum=[]):
    """
    :param trainDatadir:
    :param maxWidth:
    :param maxHeight:
    :param featureFuncsNnum: [function (returns a list of values), number of features]
    :return:
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

    # We'll rescale the images
    imageSize = maxWidth * maxHeight
    num_rows = numberofImages # one row for each image in the training dataset
    num_features = imageSize + sum([t[1] for t in featureFuncsNnum])

    # X is the feature vector with one row of features per image
    # consisting of the pixel values and our metric
    X = np.zeros((num_rows, num_features), dtype=float)
    # y is the numeric class label
    y = np.zeros(num_rows)

    files = []

    # Generate training data
    i = 0
    label = 0

    # List of string of class names
    namesClasses = list()

    report = [int((j+1)*num_rows/100.) for j in range(100)]

    print "Reading images"

    # Navigate through the list of directories
    for classFolder in directory_names:

        # Append the string class name for each class
        currentClass = classFolder.split(os.sep)[-1]
        namesClasses.append(currentClass)

        print 'Class:', currentClass

        hasImg = False
        for imgFname in os.walk(classFolder).next()[2]:

            if imgFname[-4:] != ".jpg":
                continue

            # Read in the images and create the features
            hasImg = True
            nameFileImage = "{0}{1}{2}".format(classFolder, os.sep, imgFname)
            image = imread(nameFileImage, as_grey=True)
            files.append(nameFileImage)

            # print 'Resizing from %ix%i to %ix%i' % (image.shape[0], image.shape[1], maxPixel, maxPixel)
            image = resize(image, (maxWidth, maxHeight))

            # Store the rescaled image pixels and the axis ratio
            curRow = np.reshape(image, (1, imageSize))

            #  add additional features
            for func, _ in featureFuncsNnum:
                curRow = np.append(curRow, func(image))
                # axisratio = get_minor_major_ratio(image)
                # X[i, imageSize] = axisratio

            X[i, :] = curRow

            # Store the classlabel
            y[i] = label
            i += 1

            # report progress for each 5% done
            if i in report:
                print np.ceil(i *100.0 / num_rows), "% done"

        if hasImg:
            label += 1

    return X, y, namesClasses, files


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


if __name__ == '__main__':

    maxWidth = 40
    maxLength = 40
    X_train, y, namesClasses, files = read_training_data(os.path.join(DATA_DIR, 'train'),
                                                         maxWidth, maxLength, [(get_minor_major_ratio, 1)])
    X_test, testFnames = read_test_data(os.path.join(DATA_DIR, 'test'),
                                        maxWidth, maxLength, [(get_minor_major_ratio, 1)])

    np.savetxt(os.path.join(DATA_DIR, 'XwithRatios_%i_%i.csv' % (maxWidth, maxLength)),
               X_train, delimiter=',')
    np.savetxt(os.path.join(DATA_DIR, 'y.csv'), y, delimiter=',')
    pandas.DataFrame(X_test, index=testFnames).\
        to_csv(os.path.join(DATA_DIR, 'XwithRatios_test_%i_%i.csv' % (maxWidth, maxLength)), header=False)

    print "CV-ing"
    clf = RandomForestClassifier(n_estimators=100, n_jobs=7)
    scores = cross_validation.cross_val_score(clf, X_train, y, cv=5, n_jobs=7)

    print "Accuracy of all classes"
    print np.mean(scores)

    # Get the probability predictions for computing the log-loss function
    # prediction probabilities number of samples, by number of classes
    y_pred = y * 0
    y_pred_mat = np.zeros((len(y), len(set(y))))

    for trainInd, testInd in KFold(y, n_folds=5):
        clf = RandomForestClassifier(n_estimators=100, n_jobs=3)
        clf.fit(X_train[trainInd, :], y[trainInd])

        y_pred[testInd] = clf.predict(X_train[testInd, :])
        y_pred_mat[testInd] = clf.predict_proba(X_train[testInd, :])

    print '>>>>>> Classification Report'
    print classification_report(y, y_pred, target_names=namesClasses)

    print '\n>>>>>>>Multi-class Log Loss =', multiclass_log_loss(y, y_pred_mat)

    # make predictions and write to file
    clf = RandomForestClassifier(n_estimators=100, n_jobs=7)
    clf.fit(X_train, y)
    y_test_pred = clf.predict_proba(X_test)
    pandas.DataFrame(y_test_pred, index=testFnames).to_csv(os.path.join(DATA_DIR, 'submissions', 'base.csv'),
                                                           header=False)