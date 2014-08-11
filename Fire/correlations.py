import pandas
import numpy as np

from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA

from utilities import process_data, make_column_2D
from globalVars import *


def calculate_y_corrs(xs, ys):
    """
    caclulate the correlations of each of xs' columns with y
    :param xs: 2d numpy array
    :param ys: 1d numpy array
    :return: correlations with y, pvals of the aforementioned correlations (both vectors)
    """
    temp = [pearsonr(xs[:,i], ys) for i in range(xs.shape[1])]

    y_corrs = np.abs(np.array([t[0] for t in temp]))
    y_pVals = np.array([t[1] for t in temp])

    return y_corrs, y_pVals


def calculate_feature_corrs(xs):
    """
    correlations between variables
    :param xs: 2d numpy array
    :param ys: 1d numpy array
    :return: res_corrs, res_pVals (both 2d numpy arrays)
    """

    numFeatures = xs.shape[1]

    res_corrs = np.zeros((numFeatures, numFeatures))
    res_pVals = np.zeros((numFeatures, numFeatures))

    for i in range(numFeatures):
        for j in range(numFeatures):
            corr, p = pearsonr(xs[:,i], xs[:,j])
            res_corrs[i, j] = corr
            res_pVals[i, j] = p

    # pandas.DataFrame(res_corrs).to_csv('/home/jj/code/Kaggle/Fire/corrs.csv', index=False)
    # pandas.DataFrame(res_pVals).to_csv('/home/jj/code/Kaggle/Fire/corrs_pVals.csv', index=False)

    return res_corrs, res_pVals


def plot_correlations(data, title):
    plt.matshow(data)
    plt.colorbar()
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    x_train, y_train, _, columns, weights = \
        process_data('/home/jj/code/Kaggle/Fire/Data/train.csv',
                     impute=True, imputeDataDir='/home/jj/code/Kaggle/Fire', imputeStrategy='median')

    # ----------------- correlations between variables
    # res_corrs, res_pVals = calculate_feature_corrs(x_train, y_train)
    # plot_correlations(res_corrs, 'Correlations')
    # plot_correlations(res_pVals, 'P-Values')
    #
    # # ---------- correlations between top x variables and y
    y_corrs, y_pVals = calculate_y_corrs(x_train, y_train)

    ord = y_corrs.argsort()[::-1]
    plot_correlations(make_column_2D(y_corrs[ord][:30]), 'Correlation of Top Original X Features with Y')

    # --------------- PCA
    pca = PCA(n_components=19)
    newx = pca.fit_transform(x_train, y_train)
    y_corrs, y_pVals = calculate_y_corrs(newx, y_train)
    ord = y_corrs.argsort()[::-1]
    plot_correlations(make_column_2D(y_corrs[ord]), 'Correlation of PCA Vectors with Y')
    plot_correlations(make_column_2D(y_pVals[ord]), 'P-Values of PCA Vectors')
