__author__ = 'jjin'

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble.gradient_boosting import LeastSquaresError
from sklearn.neighbors import KNeighborsClassifier

from utilities import MissingValueFiller, Normalizer
from GradientBoost_JJ import GradientBoost_JJ

rootdir = 'H:/ff/Kaggle/titanic'
# rootdir = 'C:/code/Kaggle/titanic'

svc_f =SVC(kernel='rbf', gamma=0.1, C=1.0, degree=2, tol=1e-5)
svc_m =SVC(kernel='rbf', gamma=0, C=1.0, degree=2, tol=0.1)
rf_f = RandomForestClassifier(max_features=1, n_estimators=500, random_state=100)
rf_m = RandomForestClassifier(max_features='auto', n_estimators=100, random_state=100)
gb = GradientBoostingClassifier(learning_rate=0.01, max_depth=2, n_estimators=500, subsample=0.75, random_state=100)

fillertoTry = ('filler', (MissingValueFiller(), {'method': ['mean', 'median', -1]}))
normalizerToTry = ('normalizer',(Normalizer(), {'method':['standardize', 'rescale']}))
# fillertoTry = ('filler', (MissingValueFiller(), {'method': ['mean']}))
# normalizerToTry = ('normalizer',(Normalizer(), {'method':['standardize']}))
classifiersToTry = {
    'svcsimple': (SVC(kernel='rbf'), {'tol':[1e-3, 0.1]}),
    'gradientBoostsimple': (GradientBoostingClassifier(verbose=2, n_estimators=5, subsample=0.5),
                            {'subsample':[1.0]}),
    'svc': (SVC(),
            {'kernel':['rbf', 'poly', 'sigmoid'], 'gamma':[0.0, 0.1], 'degree': [2,3,4,5],
             'tol':[1e-5, 1e-3, 0.1], 'C':[0.1,  0.5, 1.0, 5], 'shrinking':[True, False]}),
    'randomForest': (RandomForestClassifier(),
                     {'n_estimators':[200, 500, 1000], 'max_features':[1, None,'sqrt', 'auto','log2']}),
    'kneighbors': (KNeighborsClassifier(),
                   {'n_neighbors':[1, 2, 3, 5, 10, 20], 'weights':['uniform', 'distance'],
                    'algorithm':['auto','ball_tree', 'kd_tree', 'brute']}),
    'gradientBoost': (GradientBoostingClassifier(),
                      {'learning_rate': [0.005, 0.01], 'n_estimators': [500, 2500], 'max_depth': [2, 3],
                       'subsample': [0.5, 0.75, 1.0]}),
    'JJGradientBoost': (GradientBoost_JJ(learners=[svc_f, rf_m], verbosity=1, n_jobs=10,
                                         numIterations=200, lossFunction=LeastSquaresError(1), numDeemedStuck=10, randomState=99),
                        {
                            'learningRate': [0.025, 0.05, 0.1, 0.3, 0.5, 0.7], 'stoppingError': [0.05, 0.07, 0.1, 0.13], 'subsample':[0.75, 0.9, 1]
                        }),
    'JJGradientBoostsimple': (GradientBoost_JJ(learners=[svc_f, rf_f], verbosity=1, subsample=0.75,
                                               numIterations=100, lossFunction=LeastSquaresError(1), learningRate=0.7,
                                               numDeemedStuck=10, n_jobs=1, randomState=98),
                              {
                                  'stoppingError': [0.07, 0.1, 0.13]
                              })
}