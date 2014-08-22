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
import matplotlib.pyplot as plt


fname = '/home/jj/code/Kaggle/Fire/Data/6fieldsTrain.csv'
df = pandas.read_csv(fname)
numRows = df.shape[0]

# for col in df.columns:
#     print '\n------', col
#
#     try:
#         numNans = np.isnan(df[col]).sum()
#         print "Num NaN's:", numNans, 100. * numNans / numRows
#     except:
#         pass
#
#     try:
#         numZs = (df[col]=='Z').sum()
#         print "Num Zs:", numZs, 100. * numZs / numRows
#     except:
#         pass



x_train, y_train, _, columns_train, weights, y_class = \
    process_data(fname, impute=True, imputeDataDir='/home/jj/code/Kaggle/Fire/intermediateOutput', imputeStrategy='median')

clf = Ridge(alpha=1, normalize=False)
clf.fit(x_train, y_train, sample_weight=weights)
pprint(dict(zip(columns_train, clf.coef_*1e5)))


colInd = list(columns_train).index('var13')

# ---------- by category
# for val in np.unique(x_train[:, colInd]):
# for curInd in [x_train[:, colInd]==0, np.logical_not(x_train[:, colInd]==0)]:
#
#     print curInd.sum()
#     # curInd = x_train[:,colInd]==val
#     curX = x_train[curInd, :]
#     curY = y_train[curInd]
#     curWeights = weights[curInd]
#     # print '\n-----', val, curX.shape, len(curY)
#
#     clf = Ridge(alpha=1, normalize=False)
#     clf.fit(curX, curY, sample_weight=curWeights)
#
#     print 'intercept =', clf.intercept_
#     pprint(dict(zip(columns_train, clf.coef_*1e5)))


# -------- linear plots
colDict = dict([(c, list(columns_train).index(c)) for c in columns_train])

# x_train = x_train[:100000, :]
# y_train = y_train[:x_train.shape[0]]
numCols = x_train.shape[1]



# for geodem24Val in np.unique(x_train[:, colDict['geodemVar24']]):

    # ind1 = x_train[:, colDict['geodemVar24']] == geodem24Val

for var8Val in np.unique(x_train[:, colDict['var8']]):

    ind2 = x_train[:, colDict['var8']] == var8Val

        # curInd = np.logical_and(ind1, ind2)
    # curInd = ind1
    curInd = ind2

        # print '-----', geodem24Val, var8Val, curInd.sum()
    # print '-----', geodem24Val, curInd.sum()
    print '-----', var8Val, curInd.sum()

    plt.figure(figsize=(16, 16))


    # for i, col in enumerate(set(columns_train)-{'geodemVar24', 'var8'}):
    # for i, col in enumerate(set(columns_train)-{'geodemVar24'}):
    for i, col in enumerate(set(columns_train)-{'var8'}):

        colInd = colDict[col]
        print col
        plt.subplot(2, 3, i + 1)
        plt.scatter(x_train[curInd, colInd], y_train[curInd], s=3)
        # plt.title()
        plt.xlabel(col)
        plt.ylabel('Y')

    # plt.suptitle('geodem24 = %.2f; var8 = %.2f' % (geodem24Val, var8Val))
    # plt.suptitle('geodem24 = %.2f' % geodem24Val)
    plt.suptitle('var8 = %.2f' % var8Val)
    plt.show()






ind = np.array(df2['var12'])==3.0943470209
x = np.array(df['var13'])[ind]
y = y_train[ind]
plt.scatter(x,y,s=3)
plt.show()

(y_train[np.logical_not(ind)]==0).sum() *100. / (ind.sum())

plt.scatter(df2['var4'], y_train, s=3)
plt.show()


bigx, _, _, columns_train, _, _ = process_data('/home/jj/code/Kaggle/Fire/Data/train.csv',
             impute=True, imputeDataDir='/home/jj/code/Kaggle/Fire/intermediateOutput', imputeStrategy='median')[0]
fullDF = pandas.DataFrame(bigx, columns=columns_train)

ind = np.logical_and(y_train>0, y_train<10)
plt.scatter(1/weights[ind],y_train[ind],s=3);plt.show()

plt.scatter(1/weights, y_train, s=3); plt.show()