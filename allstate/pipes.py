from globalVars import *

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC

from Kaggle.utilities import makePipe


def make_pipes():
    """ makes a bunch of single-classifier pipes with corresponding pipes for calibration purposes
    """

    return {
        'GBC': makePipe([('GBC', (GradientBoostingClassifier(),
                                  {'learning_rate': [0.01, 0.1, 0.5, 1],
                                   'n_estimators': [5, 10, 25, 50, 100],
                                   'subsample': [0.7, 0.85, 1]}))]),

        'RF': makePipe([('RF', (RandomForestClassifier(n_jobs = N_JOBS),
                                {'n_estimators': [5, 10, 25, 50, 100],
                                 'max_features': [3, 0.7, 'auto', 'log2', None]
                                }))]),

        'SVC': makePipe([('SVC', (SVC(),
                                  {'C': [0.01, 0.1, 0.25, 0.5, 1],
                                   'kernel': ['rbf', 'linear', 'poly', 'rbf', 'sigmoid'],
                                   'gamma': [0, 1, 10],
                                   'shrinking': [True, False],
                                   'tol': [1e-5, 1e-3, 0.1]
                                  }))]),

        'LR': makePipe([('LR', (LogisticRegression(),
                                {'penalty': ['l1', 'l2'],
                                 'C': [0.01, 0.1, 0.5, 1, 3, 10]
                                }))]),

        # 'SGD': makePipe([('SGD', (SGDClassifier(n_jobs=20),
        #                           {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
        #                            'alpha': [1e-5, 1e-4, 1e-3, 0.01, 0.1],
        #                            'learning_rate': ['constant', 'optimal', 'invscaling']
        #                           }))])
    }