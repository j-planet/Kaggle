import sys
sys.path.append('/home/jj/code/Kaggle/allstate')

from Kaggle.utilities import print_missing_values_info, jjcross_val_score

from sklearn.preprocessing import Imputer, Normalizer, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error

from globalVars import *
from helpers import *


# --------- read data ---------
partname = 'smallTrain'
# origDataFpath = os.path.join(os.path.dirname(__file__), 'Data', partname + '.csv')
# outputFpath = os.path.join(os.path.dirname(__file__), 'Data', 'condensed_' + partname + '.csv')
origDataFpath = '/home/jj/code/Kaggle/allstate/Data/' + partname + '.csv'
outputFpath = '/home/jj/code/Kaggle/allstate/Data/' + 'condensed_' + partname + '.csv'

countTable, inputTable, outputTable, condensedTable = condense_data(origDataFpath, outputFpath, isTraining=True, verbose=False)

pdf(inputTable)

del inputTable[u'location']

# --------- impute and normalize ---------
X = Normalizer().fit_transform(Imputer().fit_transform(inputTable))  # TODO: better imputation

# plot_feature_importances(X, outputTable, inputTable.columns)

for col in outputTable.columns:
    y = outputTable[col]
    clf = GradientBoostingClassifier(subsample=0.7, n_estimators=50, learning_rate=0.1)

    print col, (1-jjcross_val_score(clf, X, y, mean_absolute_error, cv=5, n_jobs=20).mean())*100