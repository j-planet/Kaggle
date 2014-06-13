from os.path import join
import pandas
from pprint import pprint
import numpy as np

from Kaggle.utilities import RandomForester, print_missing_values_info
from globalVars import *

from sklearn.ensemble import GradientBoostingClassifier



trainHistory = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/trainHistory_wDateFields.csv")
transactions = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/transactions_train_compressed.csv")
offers = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/offers.csv")

# join train history, offers and transactions
historyOffers = pandas.merge(trainHistory, offers, left_on='offer', right_on='offer', how='left')
historyOffersTrans = pandas.merge(historyOffers, transactions, left_on='id', right_on='id', how='left')


def join_3_files(historyFpath, offersFpath, compressedTransFpath, isTrain, xFields):
    """
    joins history, offers, and compressed transactions data
    @returns: X (pandas dataframe), Y_repeater (0 and 1's), Y_numRepeats (None if hasY is False)
    """

    data = pandas.merge(pandas.merge(pandas.read_csv(historyFpath),
                                     pandas.read_csv(offersFpath),
                                     left_on='offer', right_on='offer', how='left'),
                         pandas.read_csv(compressedTransFpath),
                         left_on='id', right_on='id', how='left')

    if isTrain:
        # remove lines with inf from training data
        data = data[(data==np.inf).sum(axis=1) == 0]

        # extract Y values
        Y_numRepeats = data['repeattrips']
        Y_repeater = Y_numRepeats > 0

        # check that repeater and numRepeats are consistent
        assert sum(data['repeater'][Y_numRepeats > 0] == 'f') == 0
    else:
        Y_repeater = Y_numRepeats = None

    return data[xFields], Y_repeater, Y_numRepeats


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

    res = pandas.DataFrame({'id': X.id, 'repeatProbability': clf.predict_proba(X)[:, 1]})
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

    xFields = [f for f in historyOffersTrans.columns if f not in ['id', 'offer', 'repeattrips', 'repeater', 'offerdate']]

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