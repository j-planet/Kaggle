from helpers import *
from Kaggle.utilities import makePipe, Normalizer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

def prepPipes(simple):
    """
    the preprocessing pipes
    @param simple: whether a small range of parameters are desired
    @return: pipe, params
    """

    imputerToTry = ('filler', (Imputer(strategy='mean'), {})) if simple else \
        ('filler', (Imputer(), {'strategy': ['mean', 'median', 'most_frequent']}))

    normalizerToTry = ('normalizer', (Normalizer(), {})) if simple else \
        ('normalizer', (Normalizer(), {'method': ['standardize', 'rescale']}))

    return makePipe([imputerToTry, normalizerToTry])


def classifierPipes(simple):
    """
    the classification pipes
    @param simple: whether a small range of parameters are desired
    @return: pipe, params
    """

    pcaReducerToTry_class = ('PCAer', (PcaEr(total_var=0.85), {'whiten': [True, False]})) if simple else \
        ('PCAer', (PcaEr(total_var=0.999), {'total_var': [0.85, 0.9], 'whiten': [True, False]}))

    rfReducerToTry_class = ('RFer', (RandomForester(n_estimators=5, num_features=99), {'num_features':[15, 25]})) if simple else \
        ('RFer', (RandomForester(num_features=999, n_estimators=999), {'num_features':[0.5, 15, 25], 'n_estimators':[5, 10, 25]}))

    classifierToTry = ('GBC', (GradientBoostingClassifier(subsample=0.7, n_estimators=25),
                               {'learning_rate': [0.1, 0.5]})) if simple else \
        ('GBC', (GradientBoostingClassifier(),
                 {'max_features': ['auto', 'sqrt', 'log2'], 'learning_rate': [0.01, 0.1, 0.5, 1],
                  'n_estimators': [5, 10, 25, 50, 100], 'subsample': [0.7, 0.85, 1], 'max_depth': [3, 5, 7]}))

    # return makePipe([pcaReducerToTry_class, rfReducerToTry_class, classifierToTry])
    return makePipe([rfReducerToTry_class, classifierToTry])


def regressorPipes(simple):
    """
    the regressor pipes
    @param simple: whether a small range of parameters are desired
    @return: pipe, params
    """

    pcaReducerToTry_reg = ('PCAer', (PcaEr(total_var=0.85), {'whiten': [True, False]})) if simple else \
        ('PCAer', (PcaEr(total_var=0.999), {'total_var': [0.85, 0.9], 'whiten': [True, False]}))

    rfReducerToTry_reg = ('RFer', (RandomForester(n_estimators=5, num_features=99), {'num_features':[15, 25]})) if simple else \
        ('RFer', (RandomForester(num_features=999, n_estimators=999), {'num_features':[0.5, 15, 25], 'n_estimators':[5, 10, 25]}))

    regressorToTry = ('GBR', (GradientBoostingRegressor(loss='lad', max_features='auto', learning_rate=0.1, n_estimators=5, subsample=0.7),
                              {'learning_rate': [0.1, 0.5]})) if simple else \
        ('GBR', (GradientBoostingRegressor(loss='lad'),
                 {'max_features': ['auto', 'sqrt', 'log2'], 'subsample': [0.7, 0.85, 1], 'learning_rate': [0.01, 0.1, 0.5, 1],
                  'max_depth': [3, 5, 7], 'n_estimators': [5, 10, 25, 50, 100]}))

    # return makePipe([pcaReducerToTry_reg, rfReducerToTry_reg, regressorToTry])
    return makePipe([rfReducerToTry_reg, regressorToTry])

