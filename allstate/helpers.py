import pandas
import numpy as np
from copy import copy

from globalVars import *
from Kaggle.utilities import RandomForester

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import clone


def pdf(df, head=5):
    """
    print pandas data frame
    @param head: if None, print the whole thing, else print the first head number of rows
    """

    if head is None:
        print df.to_string()
    else:
        print df.head(n=head).to_string()


def condense_data(origDataFpath, isTraining, outputFpath = None, verbose=False):
    """
    condenses the data and outputs to file
    @param origDataFpath: path to the original data provided by Kaggle
    @param isTraining: whether the data is training data
    @param verbose: whether to print the first few lines of each table
    @param outputFpath: path to the output file, containing the condensed (i.e. one row per customer) data. Does not
                        write to file if None.

    @return: countDF, inputDF, outputTable, combinedTable
    """

    origData = pandas.read_csv(origDataFpath)
    print 'Done original reading.'

    if verbose:
        print '\n------- original data ---------'
        pdf(origData)

    inputData = origData[origData.record_type == 0]

    # ------------- encode discrete data -----------------
    # continuize car value
    inputData['car_vv'] = -1
    for k, v in CAR_VALUE_MAPPING.iteritems():
        inputData.car_vv[inputData.car_value == k] = v

    if verbose:
        print '\n------- inputData data (after car value mapping) ---------'
        pdf(inputData)

    # ------------- consolidate into one row per customer -----------------
    ind_col = u'customer_ID'
    changing_cols = [u'shopping_pt', u'record_type', u'time', u'state', u'location', u'car_value'] + OUTPUT_COLS    # TODO: handle state and location properly
    val_cols = [c for c in inputData.columns if c != ind_col]     # all columns except customer ID
    const_cols = [c for c in val_cols if c not in changing_cols]    # columns that are supposedly uniq per customer
    cids = inputData.customer_ID.unique()

    countDF = pandas.pivot_table(inputData, rows = ind_col, values=val_cols, aggfunc = lambda vec: len(vec.unique()))[val_cols]
    if verbose:
        print '\n------- countDF ---------'
        pdf(countDF)

    # average
    avgDF = pandas.pivot_table(inputData, rows = ind_col, values=const_cols + OUTPUT_COLS,
                               aggfunc=lambda vec: np.mean(vec[vec > -1]), dropna=False)[const_cols + OUTPUT_COLS]

    # last row
    lastDF = pandas.pivot_table(inputData, rows = ind_col, values = OUTPUT_COLS,
                                aggfunc = lambda vec: vec.iloc[len(vec)-1], dropna=False)[OUTPUT_COLS]
    lastDF.columns = [c + '_last' for c in OUTPUT_COLS]

    # cheapest options looked at
    temp = pandas.pivot_table(inputData, rows = ['customer_ID', 'cost'], values=OUTPUT_COLS)
    cheapDF = pandas.DataFrame(index = cids, columns=OUTPUT_COLS)

    for cid in cids:
        s = temp.ix[cid]
        cheapDF.ix[cid] = s.ix[min(s.index)]

    cheapDF = cheapDF[OUTPUT_COLS]
    cheapDF.columns = [c + '_cheap' for c in OUTPUT_COLS]

    # num of options looked at
    numOptDF = copy(countDF)[OUTPUT_COLS]
    numOptDF.columns = [c + '_num' for c in OUTPUT_COLS]

    inputDF = avgDF.join(lastDF).join(numOptDF).join(cheapDF)

    if verbose:
        print '\n------- avgDF ---------'
        pdf(avgDF)

    # ------------- output data frame -----------------
    outputTable = None
    if isTraining:
        outputTable = origData[origData.record_type==1][[ind_col] + OUTPUT_COLS]
        outputTable.columns = [ind_col] + [c + '_res' for c in OUTPUT_COLS]
        outputTable = outputTable.set_index(ind_col)
        if verbose:
            print '\n------- outputTable ---------'
            pdf(outputTable)

        combinedTable = inputDF.join(outputTable)
    else:
        combinedTable = inputDF

    if verbose:
        print '\n------- combinedTable ---------'
        pdf(combinedTable)

    # write to file
    if outputFpath is not None:
        combinedTable.to_csv(outputFpath)

    return countDF, inputDF, outputTable, combinedTable


def plot_feature_importances(X, outputTable, labels, numEstimators = 50, numTopFeatures = 7):
    """
    plots feature importances using random forest
    @param X: X data
    @param outputTable: a pandas data frame with columns being A,..,G
    @param labels: column labels
    @param numEstimators: n_estimators to use for RF
    @param numTopFeatures: number of features to show
    """

    for col in outputTable.columns:
        print '-'*10, col
        y = outputTable[col]

        rf = RandomForester(num_features = X.shape[1], n_estimators = numEstimators)
        rf.fit(X, y)
        rf.plot(num_features=numTopFeatures, labels=labels)
        print rf.top_indices(labels=labels)[1]


class CombinedClassifier(BaseEstimator, TransformerMixin):

    def __init__(self, clfs):
        self.clfs = clfs

    def fit(self, X, y):
        """
        @param y: an array of strings such as '0100122'
        """
        assert len(y[0]) == len(self.clfs), "outputTable must have the same num of columns as len(self.clfs)"

        for col, clf in enumerate(self.clfs):
            curY = np.array([int(s[col]) for s in y])
            clf.fit(X, curY)

    def predict(self, X):

        return self.combine_outputs(np.array([clf.predict(X) for clf in self.clfs]).transpose())


    @staticmethod
    def combine_outputs(outputTable):
        """
        @param outputTable: a 2D numpy array
        @return a numpy array of strings
        """

        numRows = outputTable.shape[0]
        numCols = outputTable.shape[1]

        return np.array([''.join([str(outputTable[i, l]) for l in np.arange(0, numCols)])
                for i in np.arange(0, numRows)])


    @staticmethod
    def create_by_cloning(clf, num):
        """
        create a CombinedClassifier by cloning multiple copies of one single classifer
        @return a CombinedClassifier
        """
        
        res = []
        
        for _ in np.arange(0, num):
            res.append(clone(clf))
            
        return CombinedClassifier(res)