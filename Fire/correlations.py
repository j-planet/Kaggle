import pandas
import numpy as np
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

from utilities import process_data
from globalVars import *

x_train, y_train, _, columns_train, weights = \
    process_data('/home/jj/code/Kaggle/Fire/Data/train.csv',
                 impute=True, imputeDataDir='/home/jj/code/Kaggle/Fire', imputeStrategy='median')

res_corrs = np.zeros((x_train.shape[1], x_train.shape[1]))
res_pVals = np.zeros((x_train.shape[1], x_train.shape[1]))

for i in range(x_train.shape[1]):
    for j in range(x_train.shape[1]):
        corr, p = pearsonr(x_train[:,i], x_train[:,j])
        res_corrs[i, j] = corr
        res_pVals[i, j] = p

pandas.DataFrame(res_corrs).to_csv('/home/jj/code/Kaggle/Fire/corrs.csv', index=False)
pandas.DataFrame(res_pVals).to_csv('/home/jj/code/Kaggle/Fire/corrs_pVals.csv', index=False)

plt.matshow(res_corrs)
plt.colorbar()
plt.title("Correlations")

plt.matshow(res_pVals)
plt.colorbar()
plt.title("P-Values")
plt.show()