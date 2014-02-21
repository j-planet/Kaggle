from helpers import make_data, RandomForester
from BinThenReg import *

# trainX, trainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modSmallTrain.csv",
#                                              selectFeatures=False, enc=None)

trainX, trainY, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTrain.csv",
                                           selectFeatures=False, enc=None)

prepPipe, _ = prepPipes(simple=True, useImputer=True, useNormalizer=True)

newX = prepPipe.fit_transform(trainX, trainY)

num_features = 15
rfReducer = RandomForester(num_features=num_features, n_estimators=50)
rfReducer.fit(newX, trainY)
print 'Important features:', trainX.columns[rfReducer.top_indices(num_features)]
rfReducer.plot(labels=trainX.columns)