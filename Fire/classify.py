import sys
sys.path.extend(['/home/jj/code/Kaggle/Fire'])

import pandas
import numpy as np
from pprint import pprint
from multiprocessing import cpu_count

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from Kaggle.utilities import plot_histogram, plot_feature_importances, jjcross_val_score
from globalVars import *
from evaluation import normalized_weighted_gini
from utilities import process_data
from correlations import *



x_train, y_regress, _, columns_train, weights, y_class = \
    process_data('/home/jj/code/Kaggle/Fire/Data/train.csv',
                 impute=True, imputeDataDir='/home/jj/code/Kaggle/Fire/intermediateOutput', imputeStrategy='median',
                 fieldsToUse=FIELDS_CLASS_GBC_TOP100[:5])

# print '==================== feature importances =================='
# plot_feature_importances(x_train, y_class, columns_train, numTopFeatures=0.95, numEstimators=50, num_jobs=11)

print '==================== CV =================='
# clf = GradientBoostingClassifier(learning_rate=0.1, loss='deviance')
clf = RandomForestClassifier(n_estimators=50, n_jobs=cpu_count()-2)
# clf.fit(x_train, y_class)
jjcross_val_score(clf, x_train, y_class, roc_auc_score,
                  KFold(len(y_class), n_folds=5, shuffle=True, random_state=0),
                  weights=weights)#, n_jobs=1)