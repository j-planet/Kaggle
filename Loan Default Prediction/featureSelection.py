from helpers import make_data
from BinThenReg import *
from Kaggle.utilities import RandomForester

# trainX, trainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modSmallTrain.csv", enc=None)

trainX, trainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTrain.csv", enc=None)

prepPipe, _ = prepPipes(simple=True, useImputer=True, useNormalizer=True)

newX = prepPipe.fit_transform(trainX, trainY)

num_features = 25
rfReducer = RandomForester(num_features=num_features, n_estimators=25, n_jobs=10)
rfReducer.fit(newX, trainY)
print 'Important features:', trainX.columns[rfReducer.top_indices(num_features)[0]]
rfReducer.plot(labels=trainX.columns)