import pandas
import numpy as np

from Kaggle.utilities import RandomForester, print_missing_values_info

from sklearn.preprocessing import Imputer, Normalizer, LabelEncoder

CAR_VALUE_MAPPING = dict(zip(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], np.arange(1,10)))


def pdf(df, head=5):
    """
    print pandas data frame
    @param head: if None, print the whole thing, else print the first head number of rows
    """

    if head is None:
        print df.to_string()
    else:
        print df.head(n=head).to_string()


def condense_data(origDataFpath, outputFpath, isTraining, verbose=False):
    """
    condenses the data and outputs to file
    @param origDataFpath: path to the original data provided by Kaggle
    @param outputFpath: path to the output file, containing the condensed (i.e. one row per customer) data
    @param isTraining: whether the data is training data
    @param verbose: whether to print the first few lines of each table
    @return: countTable, inputTable, outputTable, combinedTable
    """

    origData = pandas.read_csv(origDataFpath)
    if verbose:
        print '\n------- original data ---------'
        pdf(origData)

    # ------------- encode discrete data -----------------
    # continuize car value
    origData['car_vv'] = -1
    for k, v in CAR_VALUE_MAPPING.iteritems():
        origData.car_vv[origData.car_value == k] = v

    if verbose:
        print '\n------- original data (after car value mapping) ---------'
        pdf(origData)

    # ------------- consolidate into one row per customer -----------------
    ind_col = u'customer_ID'
    changing_cols = [u'shopping_pt', u'record_type', u'time', u'state', u'car_value'] # TODO: handle state properly
    val_cols = [c for c in origData.columns if c != ind_col]     # all columns except customer ID
    const_cols = [c for c in val_cols if c not in changing_cols]    # columns that are supposedly uniq per customer
    output_cols = [u'A', u'B', u'C', u'D', u'E', u'F', u'G']

    countTable = pandas.pivot_table(origData, rows = ind_col, values=val_cols, aggfunc = lambda vec: len(vec.unique()))[val_cols]
    if verbose:
        print '\n------- countTable ---------'
        pdf(countTable)

    inputTable = pandas.pivot_table(origData, rows = ind_col, values=const_cols, aggfunc=lambda vec: np.mean(vec[vec>0]), dropna=False)[const_cols]
    if verbose:
        print '\n------- inputTable ---------'
        pdf(inputTable)

    # ------------- output data frame -----------------
    outputTable = None
    if isTraining:
        outputTable = origData[origData.record_type==1][[ind_col] + output_cols]
        outputTable.columns = [ind_col] + [c + '_res' for c in output_cols]
        outputTable = outputTable.set_index(ind_col)
        if verbose:
            print '\n------- outputTable ---------'
            pdf(outputTable)

        combinedTable = inputTable.join(outputTable)
    else:
        combinedTable = inputTable

    if verbose:
        print '\n------- combinedTable ---------'
        pdf(combinedTable)

    # write to file
    combinedTable.to_csv(outputFpath)

    return origData, countTable, inputTable, outputTable, combinedTable

fname = 'smallTrain'
origDataFpath = '/home/jj/code/Kaggle/allstate/Data/' + fname + '.csv'
outputFpath = '/home/jj/code/Kaggle/allstate/Data/condensed_' + fname + '.csv'

origData, countTable, inputTable, outputTable, condensedTable \
    = condense_data(origDataFpath, outputFpath, isTraining=True, verbose=False)

pdf(inputTable)

del inputTable[u'location']

newX = Normalizer().fit_transform(Imputer().fit_transform(inputTable))
labels = inputTable.columns

for col in outputTable.columns:
    print '-'*10, col
    y = outputTable[col]

    rf = RandomForester(num_features = newX.shape[1], n_estimators = 50)
    rf.fit(newX, y)
    rf.plot(num_features=7, labels=labels)
    print rf.top_indices(labels=labels)[1]

# d = pandas.read_csv(origDataFpath)
# pdf(d)

