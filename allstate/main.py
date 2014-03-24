import pandas, os
import numpy as np
from copy import copy

from Kaggle.utilities import print_missing_values_info

from sklearn.preprocessing import Imputer, Normalizer, LabelEncoder

from globalVars import *
from helpers import *

# --------- read data ---------
partname = 'smallTrain'
origDataFpath = os.path.join(os.path.dirname(__file__), 'Data', partname + '.csv')
outputFpath = os.path.join(os.path.dirname(__file__), 'Data', 'condensed_' + partname + '.csv')

countTable, inputTable, outputTable, condensedTable = condense_data(origDataFpath, outputFpath, isTraining=True, verbose=False)

pdf(inputTable)

del inputTable[u'location']

# --------- impute and normalize ---------
X = Normalizer().fit_transform(Imputer().fit_transform(inputTable))  # TODO: better imputation

plot_feature_importances(X, outputTable, inputTable.columns)


# d = pandas.read_csv(origDataFpath)
# pdf(d)
