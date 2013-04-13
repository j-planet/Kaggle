__author__ = 'yuejin'
import sys, csv
from sklearn import metrics
from pprint import pprint
import numpy as np
from numpy.core.fromnumeric import mean, var
from numpy.lib.scimath import sqrt
from time import time

__all__ = ['csv2dict', 'normalizeData', 'loadSkiesData', 'loadHalosData', 'benchmark']

def normalizeData(data, meanOnly = False):
    """
    normalize data by subtracting mean and dividing by sd per COLUMN
    Parameters:
        - data: an array
        - meanOnly: if True subtract mean only; otherwise divide by sd too
    Returns: an array with the same dimension as data
    """
    if meanOnly:
        return data-mean(data,axis=0)

    else:
        stds = sqrt(var(data, axis=0))
        stds[stds==0] = 1   # to avoid dividing by 0

        return (data - mean(data, axis=0))/stds


def csv2dict(fname, hasHeader, fieldnames=None, dataTypes=None):
    """ load a csv file as a dict of the format {col1 name: col1 values, col2 name: col2 values, ..., coln name: coln values}
        @param hasHeader if True, uses the first row as header; otherwise uses "col1",...,"col n"
    """

    reader = csv.reader(open(fname, 'rb'), delimiter=',')

    if hasHeader:
        fieldnames = reader.next()

    data = [tuple(row) for row in reader]

    if dataTypes is None:
        res = np.array(data)
    else:   # has specific data types
        if fieldnames is None:
            fieldnames = ['col'+str(i) for i in range(len(data[0]))]

        dtype = zip(fieldnames, dataTypes)
        res = np.array(data, dtype=dtype)

    return res


MAX_NUM_GALAXIES = 740
NUM_TRAINING_SKIES = 300
NUM_TESTING_SKIES = 120

def loadSkiesData(trainOrTest='train'):
    """
    loads either all training or testing skies
    @return a dictionary of {skyID: skyData}, where skyData is
    @param trainOrTest: 'train' or 'test
    """

    assert trainOrTest in ['train','test'], 'trainOrTest must be one of train or test. %s passed.' % trainOrTest

    #    fieldnames = ['GalaxyID', 'x', 'y', 'e1', 'e2']
    X = []

    numSkies = NUM_TRAINING_SKIES if trainOrTest=='train' else NUM_TESTING_SKIES
    fnamePrefix = './data/Train_Skies/Training_Sky' if trainOrTest=='train' else './data/Test_Skies/Test_Sky'

    for skyID in np.arange(1, numSkies + 1):    # loop: 1,...,300

        # read sky data
        fname = fnamePrefix + str(skyID) + '.csv'
        #    print fname
        origSkyData = csv2dict(fname, hasHeader=True, dataTypes = ['|S20'] + [np.double]*4)

        # somehow mangle all origSkyData's to have the same "length"
        # => append values with x=avg(x), y=avg(y), e1=avg(e1), e2=avg(e2)
        numExtraRowsNeeded = MAX_NUM_GALAXIES - len(origSkyData)

        avgX = origSkyData['x'].mean()
        avgY = origSkyData['y'].mean()
        avge1 = origSkyData['e1'].mean()
        avge2 = origSkyData['e2'].mean()
        skyData = np.r_[origSkyData, np.array([('fakeGalaxy', avgX, avgY, avge1, avge2) for _ in range(numExtraRowsNeeded)], dtype=origSkyData.dtype)]

        # adjust data by subtracting average
        skyData = np.c_[skyData['x']-avgX, skyData['y']-avgY, skyData['e1']-avge1, skyData['e2']-avge2]

        # "flatten" the data
        skyData = skyData.reshape(1,-1) # x_1, y_1, e1_1, e2_1, x_2, y_2, e1_2, e2_2, ...

        # add data to the training set
        X.append(skyData)

    X = np.array(X).reshape(numSkies,-1)

    return X

def loadSkiesData_2(trainOrTest='train'):
    """
    loads either all training or testing skies
    @param trainOrTest: 'train' or 'test
    """

    assert trainOrTest in ['train','test'], 'trainOrTest must be one of train or test. %s passed.' % trainOrTest

    #    fieldnames = ['GalaxyID', 'x', 'y', 'e1', 'e2']
    X = []

    numSkies = NUM_TRAINING_SKIES if trainOrTest=='train' else NUM_TESTING_SKIES
    fnamePrefix = './data/Train_Skies/Training_Sky' if trainOrTest=='train' else './data/Test_Skies/Test_Sky'

    for skyID in np.arange(1, numSkies + 1):    # loop: 1,...,300

        # read sky data
        fname = fnamePrefix + str(skyID) + '.csv'
        #    print fname
        origSkyData = csv2dict(fname, hasHeader=True, dataTypes = ['|S20'] + [np.double]*4)

        # somehow mangle all origSkyData's to have the same "length"
        # => append values with x=avg(x), y=avg(y), e1=avg(e1), e2=avg(e2)
        numExtraRowsNeeded = MAX_NUM_GALAXIES - len(origSkyData)

        avgX = origSkyData['x'].mean()
        avgY = origSkyData['y'].mean()
        avge1 = origSkyData['e1'].mean()
        avge2 = origSkyData['e2'].mean()
        skyData = np.r_[origSkyData, np.array([('fakeGalaxy', avgX, avgY, avge1, avge2) for _ in range(numExtraRowsNeeded)], dtype=origSkyData.dtype)]

        # adjust data by subtracting average
        skyData = np.c_[skyData['x']-avgX, skyData['y']-avgY, skyData['e1']-avge1, skyData['e2']-avge2]

        # "flatten" the data
        skyData = skyData.reshape(1,-1) # x_1, y_1, e1_1, e2_1, x_2, y_2, e1_2, e2_2, ...

        # add data to the training set
        X.append(skyData)

    X = np.array(X).reshape(numSkies,-1)

    return X


def loadHalosData():
    """
    load training halos
    @returns named np.array with fields
             ('SkyId', 'numberHalos', 'x_ref', 'y_ref', 'halo_x1', 'halo_y1', 'halo_x2', 'halo_y2', 'halo_x3', 'halo_y3')
    """
    fname = './data/Training_halos.csv'
    return csv2dict(fname, hasHeader=True, dataTypes=['|S20', np.int] + [np.double]*8)

def printDoneTime(t0):
    """ prints "done in ... seconds"
    @param t0: time given by time()
    """
    print 'Done in %0.3fs.' % (time() - t0)


def benchmark(clf, X_train, y_train, X_test, y_test):
    """ train, predict and run metrics on a classifier
    """
    print 80*'_'
    print 'Training: '
    print clf
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print 'train time: %0.3fs' % train_time

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print 'test time: %0.3fs' % test_time

    score = metrics.f1_score(y_test, pred)
    print 'f1-score:   %0.3f' % score

    if hasattr(clf, 'coef_'):
        print 'dimensionality: %d' % clf.coef_.shape[1]
        print 'density: %f' % density(clf.coef_)
        print


    print '--- classification report:'
    print metrics.classification_report(y_test, pred)


    print '--- confusion matrix:'
    print metrics.confusion_matrix(y_test, pred)

    print
    clf_descr = str(clf).split('(')[0]

    return clf_descr, score, train_time, test_time

# to show how it works
if __name__=='__main__':
    fname = './data/Train_Skies/Training_Sky3.csv'
    hasHeader = True
    fieldnames = None
    dataTypes = ['|S20'] + [np.double]*4

    pprint(csv2dict(fname, hasHeader=True, dataTypes=dataTypes))