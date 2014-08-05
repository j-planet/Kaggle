import sys
sys.path.append('/home/jj/code/Kaggle/ValuedShoppers')
from os.path import join
import pandas
from pprint import pprint
import numpy as np
from itertools import product

from Kaggle.utilities import print_missing_values_info, cvScores, jjcross_val_score, plot_feature_importances
from globalVars import *

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import StratifiedShuffleSplit



def join_3_files(historyFpath, offersFpath, compressedTransFpath, isTrain, impute, xFields=None):
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
    if xFields is None:
        xFields = list(set(data.columns) - set(NON_X_FIELDS))

    X = data[xFields if isTrain else ['id'] + xFields]

    # impute missing data for X
    for i in xrange(X.shape[1]):
        col = X.icol(i)
        infInds = np.logical_or(col == np.inf, col == -np.inf)

        if infInds.sum() > 0:
            print 'imputing column', i, X.columns[i], 'with', np.mean(col[np.logical_not(infInds)])

        col[infInds] = np.mean(col[np.logical_not(infInds)])

    if impute:
        X = pandas.DataFrame(Imputer().fit_transform(X), columns=X.columns)

    return X, Y_repeater, Y_numRepeats, Y_quantiles


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

    print 'Training data:'
    print_missing_values_info(X)
    X = Imputer().fit_transform(X)

    # clf = LogisticRegression()
    clf = GradientBoostingRegressor(learning_rate=0.1, loss=lossString, n_estimators=1000, subsample=0.9)

    if fit:
        clf.fit(X, y)

    return clf


def _predict(ids, X, clf, outputFpath):
    """
    predict test data and write to file
    @return predictions
    """

    print 'Prediction data:'
    print_missing_values_info(X)
    X = Imputer().fit_transform(X)

    # res = pandas.DataFrame({'id': ids, 'repeatProbability': clf.predict_proba(X)[:, 1]})
    res = pandas.DataFrame({'id': [int(id) for id in ids], 'repeatProbability': clf.predict(X)})
    res.to_csv(outputFpath, index=False)

    return res


def predict(X, clf, outputFpath):

    ids = X.id
    del X['id']

    return _predict(ids, X, clf, outputFpath)


def plot_feature_importances_this(X_train, Y_repeater, Y_numRepeats, Y_quantiles, num_features):
    """
    @param num_features: number of features for [repeater, numrepeats, quantiles]
    """

    fields_repeater = plot_feature_importances(np.array(X_train), Y_repeater, labels=X_train.columns, numTopFeatures=num_features[0], title='Repeater')[1]
    fields_numRepeats = plot_feature_importances(np.array(X_train), Y_numRepeats, labels=X_train.columns, numTopFeatures=num_features[1], title='Number of Repeats')[1]
    fields_quantiles = plot_feature_importances(np.array(X_train), Y_quantiles, labels=X_train.columns, numTopFeatures=num_features[2], title='Quantiles of Number of Repeats')[1]

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


def write_to_vw_file(outputFname, X_df, y_series):
    """
    @param X_df: pandas data frame
    @param y_series: pandas series
    """

    with open(outputFname, 'w') as f:
        for i in xrange(X_df.shape[0]):
            line = str(y_series[i]) + ' | '
            line += ' '.join([c + ':' + str(X_df[c][i]) for c in X_df.columns])

            f.write(line + '\n')


def make_submission_from_vw_pred(inputFpath, outputFpath):
    """
    @param inputFpath: a column of numbers. no header.
    """

    ids = pandas.read_csv(join(DATA_DIR, 'testIds'))
    preds = pandas.read_csv(inputFpath, header=None)

    pandas.DataFrame({'id': np.array(ids)[:, 0], 'repeatProbability': np.array(preds)[:, 0]}).to_csv(outputFpath, index=False)


def make_cvs_inds(X_train, by, groups, isDiscrete=True):
    """
    @param X_train: a dataframe
    @param by: most likely in {"category", "company", "brand"}
    @param isDiscrete: whether the "by" field is discrete
    @param groups: how the cvs are defined by values of "by". if isDiscrete, a set of sets; otherwise, a set of division points
    """

    if not isDiscrete:
        raise NotImplementedError("Dividing by continuous fields has not been implemented.")

    res = []
    for group in groups:
        ind = X_train[by].isin(group)
        res.append((ind, np.logical_not(ind)))

    return res


def print_data_distribution_by_field(df, cols):
    """
    prints out how many of each value a field has
    @param df: pandas data frame
    @param cols: a column or a list of columns
    """

    if isinstance(cols, list):
        res = []

        for p in product(*[df[col].unique() for col in cols]):
            print p
            curInd = True
            for col, v in zip(cols, p):
                temp = df[col] == v
                if not np.any(temp):    # all false
                    curInd = [False] * df.shape[0]
                    break

                curInd = np.logical_and(curInd, temp)

            res.append((p, sum(curInd)))
    else:
        res = [(c, sum(df[cols] == c)) for c in df[cols].unique()]

    res.sort(key = lambda x: x[1],  reverse=True)

    for k, v in res:
        print k, ':', v, 1.*v/df.shape[0]

    return res


def validate_classifier(clf, X, Y_train, Y_validate, cv, use_predProb_instead, scoreFunc=roc_auc_score, n_jobs=16, test_size=0.25):
    """
    @param cv: an object (list of (trainInds, testInds)) or an integer (number of folds)
    @return: list of cv scores
    """

    cvObj = StratifiedShuffleSplit(Y_validate, n_iter=cv, test_size=test_size) if isinstance(cv, int) else cv

    scores = jjcross_val_score(clf, X, Y_train, scoreFunc, cvObj, Y_validate, n_jobs=n_jobs, use_predProb_instead=use_predProb_instead)

    return scores


if __name__ == '__main__':

    # ---- read data
    X_train, Y_repeater, Y_numRepeats, Y_quantiles = join_3_files(join(DATA_DIR, "trainHistory_wDatesQuantiles.csv"),
                                                                  join(DATA_DIR, "offers_amended.csv"),
                                                                  join(DATA_DIR, "transactions_train_compressed.csv"),
                                                                  True, impute=True,
                                                                  # xFields = None)
                                                                  xFields=X_FIELDS)

    # ---- assess feature importances
    # fields_repeater, fields_numRepeats, fields_quantiles = plot_feature_importances(X_train, Y_repeater, Y_numRepeats, Y_quantiles, [0.95]*3)
    # fieldsToUse = list(set(fields_repeater + fields_numRepeats + fields_quantiles))


    # fieldsToUse = FIELDS_10
    fieldsToUse = X_train.columns
    # print 'fields to use:', len(fieldsToUse), fieldsToUse
    #
    # # ---- classify and predict
    print '========= training'
    X_train = X_train[fieldsToUse]

    # write_to_vw_file("/home/jj/code/Kaggle/ValuedShoppers/vwFiles/61fields_train", X_train, Y_repeater)

    y_train = np.array(Y_quantiles)
    # y_train = np.array(Y_numRepeats)
    # y_train = np.array(Y_repeater)
    y_val = np.array(Y_repeater)
    clf = classify(X_train, y_train, lossString='ls', fit=False)

    print '========= validating'
    validate_classifier(clf, X_train, y_train, y_val,
                        cv= make_cvs_inds(X_train, 'company', COMPANY_CV_DIVISION),
                        use_predProb_instead=False)

    # print '========= predicting'
    # X_test = join_3_files(join(DATA_DIR, "testpythonHistory_wDateFields.csv"),
    #                      join(DATA_DIR, "offers_amended_1.csv"),
    #                      join(DATA_DIR, "transactions_test_compressed.csv"),
    #                      False, impute=True, fieldsToUse)[0]
    # # write_to_vw_file("/home/jj/code/Kaggle/ValuedShoppers/vwFiles/origTest", X_test, ['']*X_test.shape[0])
    #
    # predict(X_test, clf, '/home/jj/code/Kaggle/ValuedShoppers/submissions/10fields_quantiles_gbc_ls.csv')

    # make_submission_from_vw_pred("/home/jj/code/Kaggle/ValuedShoppers/vwFiles/pred", "/home/jj/code/Kaggle/ValuedShoppers/submissions/vw21fields")
