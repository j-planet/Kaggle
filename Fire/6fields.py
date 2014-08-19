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


fname = '/home/jj/code/Kaggle/Fire/Data/6fieldsTrain.csv'
df = pandas.read_csv(fname)
numRows = df.shape[0]

for col in df.columns:
    print '\n------', col

    try:
        numNans = np.isnan(df[col]).sum()
        print "Num NaN's:", numNans, 100. * numNans / numRows
    except:
        pass

    try:
        numZs = (df[col]=='Z').sum()
        print "Num Zs:", numZs, 1.00 * numZs / numRows
    except:
        pass



x_train, y_train, _, columns_train, weights, y_class = \
    process_data(fname, impute=True, imputeDataDir='/home/jj/code/Kaggle/Fire/intermediateOutput', imputeStrategy='median')

clf = Ridge(alpha=1, normalize=True)
clf.fit(x_train, y_train, sample_weight=weights)
pprint(dict(zip(columns_train, np.abs(clf.coef_)*1e5)))