__author__ = 'yuejin'

import pylab
from time import time
import numpy as np
from pprint import pprint
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import euclidean_distances
from sklearn.pipeline import Pipeline
from utilities import loadSkiesData, loadHalosData, normalizeData, printDoneTime, benchmark

x = loadSkiesData()
y = loadHalosData()

# ------- transform data
x = normalizeData(x)

if __name__=='__main__':
    # ------- feature seleciton on x
    pca = PCA()
    pca.fit(x)
    pylab.figure()
    pylab.clf()
    pylab.plot(pca.explained_variance_)
    pylab.axis('tight')
    pylab.xlabel('n_components')
    pylab.ylabel('explained_variance_')


    # ---------- some visualization
    t0 = time()
    mds = manifold.MDS(n_components=2, max_iter=100, n_jobs=1)
    tempNewX = mds.fit_transform(euclidean_distances(x))
    pylab.figure()
    pylab.scatter(tempNewX[:,0], tempNewX[:,1], c=y['numberHalos'], cmap=pylab.cm.Spectral)
    printDoneTime(t0)

    # -------- predict the number of halos first
    numHalosVec = y['numberHalos']
    X_train, X_test, y_train, y_test = train_test_split(x, numHalosVec, test_fraction=0.2, random_state=0)

    pipe = Pipeline(steps=[('pca', PCA()), ('svc', SVC(cache_size=2000))])

    # make params dict
    svcParams = ('svc', dict(C=[1,10,1000, 5000, 10000], degree=[1,3,5,7], gamma=[0,0.25,1],
                         kernel=['linear','poly','rbf','sigmoid'], shrinking=[True, False]))
    pcaParams = ('pca', dict(n_components=[20, 50, 75, 100, 200, 300]))
    paramsDict = dict()
    for (name, d) in [svcParams, pcaParams]:
        paramsDict.update([(name + '__' + k, val) for (k,val) in d.iteritems()])

    clf = GridSearchCV(pipe, paramsDict, n_jobs=22)
    benchmark(clf, X_train, y_train, X_test, y_test)

    print '******** best params are:'
    print clf.best_params_

    pylab.show()