import numpy as np
from pprint import pprint

from sklearn.cross_validation import KFold
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import Ridge

from utilities import process_data
from globalVars import *
from evaluation import normalized_weighted_gini


# TODO: ignore weights for now :(
def scorer(clf, X, y_true):
    """Evaluate decision function output for X relative to y_true.

    Parameters
    ----------
    clf : object
    Trained classifier to use for scoring. Must have either a
    decision_function method or a predict_proba method; the output of
    that is used to compute the score.

    X : array-like or sparse matrix
    Test data that will be fed to clf.decision_function or
    clf.predict_proba.

    y_true : array-like
    Gold standard target values for X. These must be class labels,
    not decision function values.

    Returns
    -------
    score : float
    Score function applied to prediction of estimator on X.
    """

    y_pred = clf.predict(X)
    return normalized_weighted_gini(y_true, y_pred, sample_weight=np.repeat(1, len(y_true)))


def rank_features(clf, x_train, y_train, columns):
    """
    rank features with rfe
    :param clf: estimator
    :param x_train:
    :param y_train:
    :return: the fitted rfe object
    """

    print '========== rank_features ==========='
    rfe = RFE(estimator=clf, n_features_to_select=1, verbose=2)
    rfe.fit(x_train, y_train)

    pprint(np.array(columns)[rfe.ranking_-1])

    return rfe


def select_features(clf, x_train, y_train, columns, num_folds, step=1, random_state=0):
    """
    automatic tuning of the number of features selected with cross-validation.
    :param clf: estimator
    :param x_train:
    :param y_train:
    :return: the fitted rfecv object
    """
    print '================= select_features ================'
    # Create the RFE object and compute a cross-validated score.
    cvObj = KFold(len(y_train), n_folds=num_folds, shuffle=True, random_state=random_state)

    # The "accuracy" scoring is proportional to the number of correct classifications
    rfecv = RFECV(estimator=clf, step=step, cv=cvObj, scoring=scorer, verbose=2)
    rfecv.fit(x_train, y_train)

    print '------------ Results: ----------------'
    print '>>>> Optimal number of features : %d' % rfecv.n_features_
    print '>>>> grid scores:'
    pprint(rfecv.grid_scores_)
    print '>>>> ranking of columns:'
    pprint(np.array(columns)[rfecv.ranking_-1])


    # Plot number of features VS. cross-validation scores
    import matplotlib.pyplot as plt
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    return rfecv


if __name__=='__main__':
    x_train, y_train, _, columns_train, weights = \
        process_data('/home/jj/code/Kaggle/Fire/Data/train.csv',
                     impute=True, imputeDataDir='/home/jj/code/Kaggle/Fire', imputeStrategy='median',
                     fieldsToUse=None)

    clf = Ridge(alpha = 0.1)

    rank_features(clf, x_train, y_train, columns_train)
    # select_features(clf, x_train, y_train, columns_train, num_folds=5, step=0.075)
