from datetime import datetime
import numpy as np

from Kaggle.CV_Utilities import loadObject
from Kaggle.utilities import DatasetPair
from helpers import quick_score, make_data


bestPipe = loadObject('/home/jj/code/Kaggle/Loan Default Prediction/output/gridSearchOutput/logistic.pk')['best_estimator']

# ------ small
# x, y, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modSmallTrain.csv", enc=None)
# ------ full
x, y, _, enc = make_data("/home/jj/code/Kaggle/Loan Default Prediction/Data/modTrain.csv", enc=None)
data = DatasetPair(np.array(x), np.array(y))

# ---------- double-check cv score and classification roc auc
dt = datetime.now()
print 'jj score:', quick_score(bestPipe, data.X, data.Y)
print 'quick jj score took', datetime.now() - dt

dt = datetime.now()
bestPipe.classification_metrics(data.X, data.Y, n_iter=10)
print 'Classifier metrics took', datetime.now() - dt
