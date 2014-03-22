import pandas
import numpy as np

from Kaggle.utilities import RandomForester, print_missing_values_info

from sklearn.preprocessing import Imputer, Normalizer, LabelEncoder

CAR_VALUE_MAPPING = pandas.DataFrame({'car_value': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', ''],
                                      'car_vv': [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan]})


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
    @return: countTable, inputTable, outputTable, condensedTable
    """

    d = pandas.read_csv(origDataFpath)
    if verbose: print d.head().to_string()

    # ------------- encode discrete data -----------------

    # continuize car value
    d = pandas.merge(d, CAR_VALUE_MAPPING)

    # ------------- input data frames -----------------
    cols = [u'customer_ID', u'shopping_pt', u'record_type', u'day', u'time', u'state', u'location', u'group_size',
            u'homeowner', u'car_age', u'car_value', u'car_vv', u'risk_factor', u'age_oldest', u'age_youngest', u'married_couple',
            u'C_previous', u'duration_previous', u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'cost']
    changing_cols = [u'shopping_pt', u'record_type', u'time', u'state', u'car_value'] # TODO: handle state properly

    indColName = u'customer_ID'
    val_cols = [c for c in cols if c != indColName]     # all columns except customer ID

    # columns that are supposedly uniq per customer
    const_cols = [c for c in val_cols if c not in changing_cols]

    # countTable = pandas.pivot_table(d, rows = indColName, values=val_cols, aggfunc = lambda vec: len(vec.unique()))[val_cols]
    countTable = pandas.pivot_table(d, rows = indColName, values=val_cols, aggfunc = lambda vec: vec.iloc[len(vec)-1])[val_cols]
    if verbose: print countTable.head().to_string()

    inputTable = pandas.pivot_table(d, rows = indColName, values=const_cols)[const_cols]
    if verbose: print inputTable.head().to_string()


    # ------------- output data frame -----------------
    outputTable = None
    if isTraining:
        output_cols = [u'A', u'B', u'C', u'D', u'E', u'F', u'G']
        outputTable = d[d.record_type==1][[indColName] + output_cols]
        outputTable.columns = [indColName] + [c + '_res' for c in output_cols]
        outputTable = outputTable.set_index(indColName)
        if verbose: print outputTable.head().to_string()

        # ------------- combined data frame -----------------
        condensedTable = inputTable.join(outputTable)
    else:
        condensedTable = inputTable

    if verbose: print condensedTable.head().to_string()

    # write to file
    condensedTable.to_csv(outputFpath)

    return countTable, inputTable, outputTable, condensedTable

fname = 'smallTrain'
origDataFpath = '/home/jj/code/Kaggle/allstate/Data/' + fname + '.csv'
outputFpath = '/home/jj/code/Kaggle/allstate/Data/condensed_' + fname + '.csv'

countTable, inputTable, outputTable, condensedTable \
    = condense_data(origDataFpath, outputFpath, isTraining=True, verbose=False)

del inputTable[u'location']
del inputTable[u'car_vv']
del inputTable[u'homeowner']
inputTable = inputTable[['A','B','C','D','E','F','G']]

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

