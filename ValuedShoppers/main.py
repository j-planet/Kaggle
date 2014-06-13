import sys
from os.path import join
import pandas
from pprint import pprint
import numpy as np

from Kaggle.utilities import RandomForester, print_missing_values_info
from globalVars import *

from sklearn.ensemble import GradientBoostingClassifier
sys.path.append('/home/jj/code/Kaggle/ValuedShoppers')


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

        # check that repeater and numRepeats are consistent
        assert sum(data['repeater'][Y_numRepeats > 0] == 'f') == 0
    else:
        Y_repeater = Y_numRepeats = None

    # ------------- find X
    X = data[xFields if isTrain else ['id'] + xFields]

    # impute missing data for X
    for i in xrange(X.shape[1]):
        col = X.icol(i)
        infInds = np.logical_or(col == np.inf, col == -np.inf)

        if infInds.sum() > 0:
            print 'imputing column', i, X.columns[i], 'with', np.mean(col[np.logical_not(infInds)])

        col[infInds] = np.mean(col[np.logical_not(infInds)])

    return X, Y_repeater, Y_numRepeats


def classify(X, Y_repeater, Y_numRepeats):
    """
    train classifier on training data
    @return classifier
    """

    clf = GradientBoostingClassifier(learning_rate=0.01, loss='deviance', n_estimators=100, subsample=0.8)
    clf.fit(X, Y_repeater)

    # check training predictions
    temp = pandas.DataFrame({'repeattrips': Y_numRepeats,
                              'repeater': Y_repeater,
                              'pred': clf.predict(X),
                              'probs': clf.predict_proba(X)[:,1]})
    temp.to_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/checkTrainPredictions.csv", index=False)

    return clf


def predict(X, clf, outputFpath):
    """
    predict test data and write to file
    @return predictions
    """

    ids = X.id
    del X['id']

    print_missing_values_info(X)
    res = pandas.DataFrame({'id': ids, 'repeatProbability': clf.predict_proba(X)[:, 1]})
    res.to_csv(outputFpath, index=False)

    return res


def plot_feature_importances(X, Y, labels, numTopFeatures, numEstimators = 50):

    rf = RandomForester(num_features = X.shape[1], n_estimators = numEstimators)
    rf.fit(X, Y)

    topFeatureInd, topFeatureLabels, topFeatureImportances = rf.top_indices(labels=labels)

    print 'Top features:'
    pprint(np.transpose([topFeatureLabels, topFeatureImportances]))

    rf.plot(num_features=numTopFeatures, labels=labels)

    return topFeatureInd, topFeatureLabels, topFeatureImportances


if __name__ == '__main__':

    xFields = ['chain',
               'market',
               'dayOfTheWeek',
               'isWeekend',
               'isHoliday',
               'isAWeekBeforeHol',
               'isTwoWeeksBeforeHol',
               'isAWeekAfterHol',
               'isTwoWeeksAfterHol',
               'category',
               'quantity',
               'company',
               'offervalue',
               'brand',
               'chain_freq',
               'category_freq',
               'company_freq',
               'brand_freq',
               'chain_hasShopped',
               'category_hasShopped',
               'company_hasShopped',
               'brand_hasShopped',
               'daysSinceLastPch',
               'avgWeeklyPchAmt',
               'pchAmtWiWeekOfOffer']

    # ---- read data
    X_train, Y_repeater, Y_numRepeats = join_3_files(join(DATA_DIR, "trainHistory_wDateFields.csv"),
                                                     join(DATA_DIR, "offers.csv"),
                                                     join(DATA_DIR, "transactions_train_compressed.csv"),
                                                     True, xFields)

    # ---- assess feature importances
    inds_repeater = plot_feature_importances(np.array(X_train), Y_repeater, labels=X_train.columns, numTopFeatures=X_train.shape[1]/2)[0]
    inds_numRepeats = plot_feature_importances(np.array(X_train), Y_numRepeats, labels=X_train.columns, numTopFeatures=X_train.shape[1]/2)[0]

    # ---- classify and predict
    print '========= training'
    clf = classify(X_train, Y_repeater, Y_numRepeats)

    X_test = join_3_files(join(DATA_DIR, "testHistory_wDateFields.csv"),
                         join(DATA_DIR, "offers.csv"),
                         join(DATA_DIR, "transactions_test_compressed.csv"),
                         False, xFields)[0]

    print '========= predicting'
    predict(X_test, clf, '/home/jj/code/Kaggle/ValuedShoppers/submissions/25Fields.csv')