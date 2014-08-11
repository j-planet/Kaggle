import sys
sys.path.extend(['/home/jj/code/Kaggle/Fire'])

import pandas
import numpy as np
from pprint import pprint

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA

from Kaggle.utilities import plot_histogram, plot_feature_importances, jjcross_val_score
from globalVars import *
from evaluation import normalized_weighted_gini
from utilities import process_data


def gridSearch(cvOutputFname, x_train, y_train, weights, num_folds = 10):
    print '================== Grid Search for the Best Parameter  =================='

    cvOutputFile = open(cvOutputFname, 'w')
    res = {}
    cvObj = KFold(len(y_train), n_folds=num_folds, shuffle=True, random_state=0)

    for tolerance in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]:
        for alpha in np.arange(0.01, 5, 0.025):
            print '>>>> alpha=', alpha, ', tolerance =', tolerance
            clf.set_params(alpha=alpha, tol=tolerance)
            scores = jjcross_val_score(clf, x_train, y_train, normalized_weighted_gini, cvObj, weights=weights,
                                       verbose=False)
            meanScore = np.mean(scores)
            stdScore = np.std(scores)
            s = 'alpha = %f, tolerance = %f, mean = %f, std = %f\n' % (alpha, tolerance, meanScore, stdScore)
            print s
            res[(alpha, tolerance)] = (meanScore, stdScore)
            cvOutputFile.write(s)
    print '>>>>>> Result sorted by mean score:'
    pprint(sorted(res.items(), key=lambda x: -x[1][0]))
    cvOutputFile.close()

    return res


if __name__ == '__main__':

    # print 'about to plot feature importances'
    # plot_feature_importances(x_train, np.array(y_train), x_train.columns, numTopFeatures=0.85, numEstimators=50)

    x_train, y_train, _, columns_train, weights = \
        process_data('/home/jj/code/Kaggle/Fire/Data/train.csv',
                     impute=True, imputeDataDir='/home/jj/code/Kaggle/Fire', imputeStrategy='median',
                     fieldsToUse=FIELDS_CORR_ORDERED[:5])

    # clf = GradientBoostingRegressor(loss='quantile', learning_rate=0.02, n_estimators=100, subsample=0.9)
    # clf = LogisticRegression()
    clf = Ridge(alpha=0.1)
    # clf = SVR()

    # ================== CV ==================
    print '================== CV =================='

    scores = jjcross_val_score(clf, x_train, y_train, normalized_weighted_gini,
                               KFold(len(y_train), n_folds=5, shuffle=True), weights=weights)

    # ================== Grid Search for the Best Parameter ==================
    # gridSearch('/home/jj/code/Kaggle/Fire/cvRes/Ridge.txt', x_train, y_train, weights)

    # ================== train ==================
    # print '================== train =================='
    #
    # clf.fit(np.array(x_train), np.array(y_train), weights)
    #
    # # ================== predict ==================
    # print '================== predict =================='
    # x_test, _, ids_pred, _, _ = process_data('/home/jj/code/Kaggle/Fire/Data/test.csv',
    #                                          impute=True, imputeDataDir='/home/jj/code/Kaggle/Fire', imputeStrategy='median',
    #                                          fieldsToUse=columns_train)
    # pred = clf.predict(x_test)
    # pandas.DataFrame({'id': ids_pred, 'target': pred}).\
    #     to_csv('/home/jj/code/Kaggle/Fire/submissions/corrfieldsRidge.csv', index=False)
