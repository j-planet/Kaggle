import sys
sys.path.extend(['/home/jj/code/Kaggle/Fire'])

import pandas
import numpy as np
from pprint import pprint

from sklearn.linear_model import *
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.tree import DecisionTreeClassifier


from Kaggle.utilities import plot_histogram, plot_feature_importances, jjcross_val_score
from globalVars import *
from evaluation import normalized_weighted_gini
from utilities import process_data, gridSearch
from correlations import *
from ClassifyThenRegress import ClassifyThenRegress
from GroupThenRegress import GroupThenRegress


def fieldNamesToInd(allColumnNames, selectNames):
    """
    convert field names to indices in the list of all column names
    :param allColumnNames:
    :param selectNames:
    :return: numpy array
    """

    return np.array([list(allColumnNames).index(v) for v in set(selectNames) & set(allColumnNames)])

if __name__ == '__main__':



    x_train, y_train, _, columns_train, weights, y_class = \
        process_data('/home/jj/code/Kaggle/Fire/Data/train.csv',
                     impute=True, imputeDataDir='/home/jj/code/Kaggle/Fire/intermediateOutput', imputeStrategy='median',
                     fieldsToUse=['var11', 'var8', 'var13'])
                     # fieldsToUse=FIELDS_CLASS_GBC_TOP100[:30])
                     # fieldsToUse=FIELDS_CDF_CORR_TOP99[:19])
    # y_cdfs = np.array(pandas.read_csv('/home/jj/code/Kaggle/Fire/Data/y_pcdfs.csv')).reshape(NUM_TRAIN_SAMPLES,)[:len(y_train)]  # in case smallTrain is used
    # clf = GradientBoostingRegressor(loss='quantile', learning_rate=0.02, n_estimators=100, subsample=0.9)
    # clf = LogisticRegression()

    plot_feature_importances(x_train, np.array(y_train), columns_train, numTopFeatures=3, numEstimators=50)

    classifier = SVR(kernel='rbf')
    regressor = Ridge(alpha=1)
    classFields = fieldNamesToInd(columns_train, FIELDS_CLASS_GBC_TOP100[:20])
    regFields = fieldNamesToInd(columns_train, FIELDS_CORR_ORDERED_TOP99[:20])

    # clf = GroupThenRegress(list(columns_train).index('var8'),
    #                        Ridge(alpha=1, normalize=False),
    #                        verbose=1)

    # clf = SVR()

    # ================== CORRELATION ==================
    # print '================== CORRELATION =================='
    # print x_train.shape
    # numFields = 30
    # x_train, newCols = create_new_features(x_train, columns=columns_train)
    # corrs = calculate_y_corrs(x_train, y_train)[0]
    # ord = corrs.argsort()[::-1][:numFields]
    # x_train = x_train[:, ord]

    # ================== CV ==================
    print '================== CV =================='
    scores = jjcross_val_score(regressor, x_train, y_train, normalized_weighted_gini,
                               KFold(len(y_train), n_folds=5, shuffle=True, random_state=0), weights=weights)#, n_jobs=1)

    # ================== Grid Search for the Best Parameter ==================
    # gridSearch(clf, '/home/jj/code/Kaggle/Fire/cvRes/RidgeGroupThenRegress.txt', x_train, y_train, weights, innerclf=True)

    # ================== train ==================
    # print '================== train =================='
    # clf.fit(x_train, y_train, sample_weight=weights)
    # #
    # # # ================== predict ==================
    # print '================== predict =================='
    # x_test, _, ids_pred, _, _, _ = process_data('/home/jj/code/Kaggle/Fire/Data/test.csv',
    #                                          impute=True, imputeDataDir='/home/jj/code/Kaggle/Fire/intermediateOutput', imputeStrategy='median',
    #                                          fieldsToUse=columns_train)
    # pred = clf.predict(x_test)
    # pandas.DataFrame({'id': ids_pred, 'target': pred}).\
    #     to_csv('/home/jj/code/Kaggle/Fire/submissions/6FieldsGroupThenRegress_var8.csv', index=False)