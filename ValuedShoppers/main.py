import sys
sys.path.append('/home/jj/code/Kaggle/ValuedShoppers')
from os.path import join
import pandas
from pprint import pprint
import numpy as np

from Kaggle.utilities import RandomForester, print_missing_values_info, cvScores
from globalVars import *

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score



def join_3_files(historyFpath, offersFpath, compressedTransFpath, isTrain, xFields):
    """
    joins history, offers, and compressed transactions data
    handles rows with Inf values
    selects the relevant fields via xFields

    @returns: X (pandas dataframe), Y_repeater (0 and 1's), Y_numRepeats (None if hasY is False)
    """

    data = pandas.merge(pandas.merge(pandas.read_csv(historyFpath),
                                     pandas.read_csv(offersFpath),
                                     left_on='offer', right_on='offer', how='left'),
                         pandas.read_csv(compressedTransFpath),
                         left_on='id', right_on='id', how='left')

    # ------------- find Ys
    if isTrain:
        # extract Y values
        Y_numRepeats = data['repeattrips']
        Y_repeater = Y_numRepeats > 0
        Y_quantiles = data['repeatQuantiles']

        # check that repeater and numRepeats are consistent
        assert sum(data['repeater'][Y_numRepeats > 0] == 'f') == 0
    else:
        Y_repeater = Y_numRepeats = Y_quantiles = None

    # ------------- find X
    X = data[xFields if isTrain else ['id'] + xFields]

    # impute missing data for X
    for i in xrange(X.shape[1]):
        col = X.icol(i)
        infInds = np.logical_or(col == np.inf, col == -np.inf)

        if infInds.sum() > 0:
            print 'imputing column', i, X.columns[i], 'with', np.mean(col[np.logical_not(infInds)])

        col[infInds] = np.mean(col[np.logical_not(infInds)])

    return X, Y_repeater, Y_numRepeats, Y_quantiles


def cv_scores(clf, X, Y_train, Y_test, numCvs=5, n_jobs=16):
    """
    uses auc score as the score func
    @param Y_train: Y_quantiles if targeting the quantiles
    @param Y_test: Y_repeater
    """

    return cvScores(clf, X, Y_train, scoreFuncsToUse='auc_score', numCVs=5, n_jobs=n_jobs, test_size=0.25, y_test=Y_test)


def test_training_results(clf, X, Y_repeater, Y_numRepeats):
    """
    check training predictions
    """

    pred = clf.predict(X)

    temp = pandas.DataFrame({'repeattrips': Y_numRepeats,
                             'repeater': Y_repeater,
                             'pred': pred})

    temp.to_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/checkTrainPredictions.csv", index=False)

    return roc_auc_score(Y_repeater, pred)


def classify(X, y, lossString, fit=True):
    """
    train classifier on training data
    @return classifier
    """

    # clf = LogisticRegression()
    clf = GradientBoostingRegressor(learning_rate=0.001, loss=lossString, n_estimators=100, subsample=0.9)

    if fit:
        clf.fit(X, y)

    return clf


def _predict(ids, X, clf, outputFpath):
    """
    predict test data and write to file
    @return predictions
    """

    print_missing_values_info(X)

    # res = pandas.DataFrame({'id': ids, 'repeatProbability': clf.predict_proba(X)[:, 1]})
    res = pandas.DataFrame({'id': ids, 'repeatProbability': clf.predict(X)})
    res.to_csv(outputFpath, index=False)

    return res


def predict(X, clf, outputFpath):

    ids = X.id
    del X['id']

    return _predict(ids, np.array(X), clf, outputFpath)


def _plot_feature_importances(X, Y, labels, numTopFeatures, numEstimators = 50):

    rf = RandomForester(num_features = X.shape[1], n_estimators = numEstimators)
    rf.fit(X, Y)

    topFeatureInd, topFeatureLabels, topFeatureImportances = rf.top_indices(labels=labels)

    print 'Top features:'
    pprint(np.transpose([topFeatureLabels, topFeatureImportances]))

    rf.plot(num_features=numTopFeatures, labels=labels)

    return topFeatureInd, topFeatureLabels, topFeatureImportances


def plot_feature_importances(X_train, Y_repeater, Y_numRepeats, Y_quantiles):

    fields_repeater = _plot_feature_importances(np.array(X_train), Y_repeater, labels=X_train.columns, numTopFeatures=X_train.shape[1]/2)[1]
    fields_numRepeats = _plot_feature_importances(np.array(X_train), Y_numRepeats, labels=X_train.columns, numTopFeatures=X_train.shape[1]/2)[1]
    fields_quantiles = _plot_feature_importances(np.array(X_train), Y_quantiles, labels=X_train.columns, numTopFeatures=X_train.shape[1]/2)[1]

    return fields_repeater, fields_numRepeats, fields_quantiles


def compare_test_and_train_data(X_train, X_test):
    """
    checks if the test data is representitative of training data
    """

    for f in X_train.columns:
        print '------', f
        if X_FIELDS[f]: # is discrete
            d = set(X_test[f].unique()) - set(X_train[f].unique())

            if len(d) > 0:
                print len(d), 1. * sum(v in d for v in X_test[f]) / X_test.shape[0]
                pprint(d)
        else:
            minTest = X_test[f].min()
            maxTest = X_test[f].max()
            minTrain = X_train[f].min()
            maxTrain = X_train[f].max()

            if minTest < minTrain or maxTest > maxTrain:
                print 'Test range is wider: (%f, %f) vs (%f, %f)' % (minTest, maxTest, minTrain, maxTrain)
                print 'test range/ train range =', 1.*(maxTest-minTest)/(maxTrain-minTrain)


if __name__ == '__main__':

    # ---- read data
    X_train, Y_repeater, Y_numRepeats, Y_quantiles = join_3_files(join(DATA_DIR, "trainHistory_wDatesQuantiles.csv"),
                                                                  join(DATA_DIR, "offers.csv"),
                                                                  join(DATA_DIR, "transactions_train_compressed.csv"),
                                                                  True, X_FIELDS.keys())

    # ---- assess feature importances
    # fields_repeater, fields_numRepeats, fields_quantiles = plot_feature_importances(X_train, Y_repeater, Y_numRepeats, Y_quantiles)
    # fieldsToUse = list(set(fields_repeater[:5] + fields_numRepeats[:5] + fields_quantiles[:8]))
    fieldsToUse = FIELDS_17
    print 'fields to use:', len(fieldsToUse), fieldsToUse

    # ---- classify and predict
    print '========= training'
    X_train = X_train[fieldsToUse]
    y_train = np.array(Y_quantiles)
    # y_train = np.array(Y_numRepeats)
    y_val = np.array(Y_repeater)
    clf = classify(np.array(X_train), y_train, lossString='ls', fit=True)

    # print 'CV scores:', cv_scores(clf, np.array(X_train), y_train, y_val, n_jobs=16, numCvs=16)

    print '========= predicting'
    X_test = join_3_files(join(DATA_DIR, "testHistory_wDateFields.csv"),
                         join(DATA_DIR, "offers.csv"),
                         join(DATA_DIR, "transactions_test_compressed.csv"),
                         False, fieldsToUse)[0]


    predict(X_test, clf, '/home/jj/code/Kaggle/ValuedShoppers/submissions/17fields_quantiles_gbc_long.csv')

    # compare_test_and_train_data(X_train, X_test)