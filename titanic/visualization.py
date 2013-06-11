__author__ = 'jjin'


import os
from copy import deepcopy
from time import time
import numpy as np
from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap, SpectralEmbedding
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import ExtraTreesClassifier
from StringIO import StringIO
from pydot import graph_from_dot_data
from main import rootdir, readData
from titanicutilities import readTrainingData, readTestingData
from utilities import printDoneTime, Normalizer, MissingValueFiller
import pylab
from matplotlib.ticker import NullFormatter

def plotManifolds(all_train_x, all_train_y, all_test_x, trainSampleWeights, testSampleWeights, n_jobs=23):
    mult = 20
    trainSampleWeights = mult*(trainSampleWeights - trainSampleWeights.min()) + 2
    testSampleWeights = mult*(testSampleWeights - testSampleWeights.min()) + 2

    # ------ transform and plot ------
    n_neighbors = 10
    names = ['LLE']
    transformers = [('LLE', LocallyLinearEmbedding(n_neighbors=n_neighbors, method='standard')),
                    ('MDS', MDS(n_jobs=n_jobs)),
                    ('Isomap', Isomap(n_neighbors=n_neighbors)),
                    ('Spectral Embedding', SpectralEmbedding(n_neighbors=n_neighbors))]

    fig = pylab.figure()

    for i, (name, transformer) in enumerate(transformers):
        print '---', name, '---'
        # transform
        t0 = time()
        traindata = deepcopy(transformer).fit_transform(all_train_x)
        testdata = deepcopy(transformer).fit_transform(all_test_x)
        printDoneTime(t0)

        # plot
        ax = fig.add_subplot(2, len(transformers),2*i+1)
        for surv in [0,1]:
            color = "red" if surv==1 else "blue"
            label = 'survived' if surv==1 else 'died'
            ind = all_train_y==surv
            pylab.scatter(traindata[ind,0], traindata[ind,1], c=color, alpha=0.7, s=trainSampleWeights, linewidths=0.5, label=label)
        pylab.title(name + " (Training Data)")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.legend(loc="best")
        pylab.axis('tight')

        ax = fig.add_subplot(2, len(transformers),2*i+2)
        pylab.scatter(testdata[:,0], testdata[:,1], c="grey", s=testSampleWeights)
        pylab.title(name + " (Test Data)")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        pylab.axis('tight')

    pylab.show()

def drawTree(x, y, outputFname):
    # fit
    clf = DecisionTreeClassifier(compute_importances=True)
    clf.fit(x, y)
    print 'feature importances:', clf.feature_importances_

    # plot
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data)
    graph = graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(outputFname)

def plotFeatureImportances(x, y, fieldNames, numTrees = 100):
    print fieldNames
    # fit
    forest = ExtraTreesClassifier(n_estimators=numTrees, compute_importances=True, random_state=0)
    forest.fit(x, y)

    # get importances
    importances = forest.feature_importances_
    print sum(importances)
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # present
    numFeatures = len(importances)
    print 'feature ranking:'
    for i in xrange(numFeatures):
        print '%d. feature %d (%s) has importance %f' % (i+1, indices[i], fieldNames[indices[i]], importances[indices[i]])

    xtickLabels = [fieldNames[i] for i in indices]
    pylab.figure()
    pylab.title('Feature Importances From A Random Forest with %s trees' % numTrees)
    pylab.bar(xrange(numFeatures), importances[indices], color='r', yerr=std[indices], align='center')
    pylab.xticks(xrange(numFeatures), xtickLabels)
    pylab.xlim([-1, numFeatures])
    pylab.show()

if __name__=="__main__":

    # transPipe = Pipeline([('filler', MissingValueFiller()), ('normer', Normalizer())])
    transPipe = Pipeline([('filler', MissingValueFiller())])

    data = readData(rootdir)[0]
    all_x = transPipe.fit_transform(data.X)
    all_y = data.Y


    plotFeatureImportances(all_x, all_y, data.fieldNames)