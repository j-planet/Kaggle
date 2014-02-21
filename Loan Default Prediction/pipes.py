from helpers import *
from Kaggle.utilities import makePipe, Normalizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


def prepPipes(simple, useImputer=True, useNormalizer=True):
    """
    the preprocessing pipes
    @param simple: whether a small range of parameters are desired
    @return: pipe, params
    """

    imputerToTry = ('filler', (Imputer(strategy='mean'), {})) if simple else \
        ('filler', (Imputer(), {'strategy': ['mean', 'median', 'most_frequent']}))

    normalizerToTry = ('normalizer', (Normalizer(), {})) if simple else \
        ('normalizer', (Normalizer(), {'method': ['standardize', 'rescale']}))

    if useImputer and useNormalizer:
        return makePipe([imputerToTry, normalizerToTry])
    elif useImputer and not useNormalizer:
        return makePipe([imputerToTry])
    elif not useImputer and useNormalizer:
        makePipe([normalizerToTry])
    else:
        raise Exception('Have to use at least one of imputer and normalizer.')


def classifierPipes(simple, name, usePCA = True, useRF = True):
    """
    the classification pipes
    @param simple: whether a small range of parameters are desired
    @param name: which classifier to use. one of {'GBC', 'logistic'}
    @return: pipe, params
    """

    pcaReducerToTry_class = ('PCAer', (PcaEr(total_var=0.85), {'whiten': [True, False]})) if simple else \
        ('PCAer', (PcaEr(total_var=0.999), {'total_var': [0.85, 0.9], 'whiten': [True, False]}))

    rfReducerToTry_class = ('RFer', (RandomForester(n_estimators=25, num_features=99), {'num_features':[15, 25]})) if simple else \
        ('RFer', (RandomForester(num_features=999, n_estimators=999), {'num_features':[0.5, 15, 25], 'n_estimators':[25, 50]}))

    if name == 'GBC':
        classifierToTry = ('GBC', (GradientBoostingClassifier(subsample=0.7, n_estimators=25),
                                   {'learning_rate': [0.1, 0.5]})) if simple else \
            ('GBC', (GradientBoostingClassifier(),
                     {'max_features': ['auto', 'sqrt', 'log2'], 'learning_rate': [0.01, 0.1, 0.5, 1],
                      'n_estimators': [5, 10, 25, 50, 100], 'subsample': [0.7, 0.85, 1], 'max_depth': [3, 5, 7]}))
    elif name == 'logistic':
        classifierToTry = ('logistic', (LogisticRegression(penalty='l2', C=1.0),
                                        {'tol': [0.01, 0.0001]} if simple else
                                        {'penalty': ['l1', 'l2'], 'C': [0.001, 0.1, 0.25, 0.5, 0.7, 1.0], 'tol': [0.01, 0.001, 0.0001]}
        ))

    else:
        raise Exception('Classifier %s is not supported.' % name)

    if usePCA and useRF:
        return makePipe([pcaReducerToTry_class, rfReducerToTry_class, classifierToTry])
    elif usePCA and not useRF:
        return makePipe([pcaReducerToTry_class, classifierToTry])
    elif not usePCA and useRF:
        return makePipe([rfReducerToTry_class, classifierToTry])
    else:
        return makePipe([classifierToTry])


def regressorPipes(simple, usePCA = True, useRF = True):
    """
    the regressor pipes
    @param simple: whether a small range of parameters are desired
    @return: pipe, params
    """

    pcaReducerToTry_reg = ('PCAer', (PcaEr(total_var=0.85), {'whiten': [True, False]})) if simple else \
        ('PCAer', (PcaEr(total_var=0.999), {'total_var': [0.85, 0.9], 'whiten': [True, False]}))

    rfReducerToTry_reg = ('RFer', (RandomForester(n_estimators=25, num_features=99), {'num_features':[15, 25]})) if simple else \
        ('RFer', (RandomForester(num_features=999, n_estimators=999), {'num_features':[0.5, 15, 25], 'n_estimators':[25, 50]}))

    regressorToTry = ('GBR', (GradientBoostingRegressor(loss='lad', max_features='auto', learning_rate=0.1, n_estimators=5, subsample=0.7),
                              {'learning_rate': [0.1, 0.5]})) if simple else \
        ('GBR', (GradientBoostingRegressor(loss='lad'),
                 {'max_features': ['auto', 'sqrt', 'log2'], 'subsample': [0.7, 0.85, 1], 'learning_rate': [0.01, 0.1, 0.5, 1],
                  'max_depth': [3, 5, 7], 'n_estimators': [5, 10, 25, 50, 100]}))

    if usePCA and useRF:
        return makePipe([pcaReducerToTry_reg, rfReducerToTry_reg, regressorToTry])
    elif usePCA and not useRF:
        return makePipe([pcaReducerToTry_reg, regressorToTry])
    elif not usePCA and useRF:
        return makePipe([rfReducerToTry_reg, regressorToTry])
    else:
        return makePipe([regressorToTry])