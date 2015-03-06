from theano.tensor.signal import downsample

__author__ = 'JennyYueJin'


import cPickle
import os
import sys
import json
import time
import datetime
import numpy as np
import pandas
from matplotlib import pyplot as plt
from PIL import Image
from pprint import pprint
from math import log10

from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

# theano.config.openmp = True
# theano.config.optimizer = 'fast_run'
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity='high'

sys.path.extend(['/Users/jennyyuejin/K/tryTheano','/Users/jennyyuejin/K'])

from logisticRegressionExample import LogisticRegression, load_data
from mlpExample import HiddenLayer

from NDSB.fileMangling import make_submission_file
from NDSB.global_vars import CLASS_NAMES
from leNetConvPoolLayer import LeNetConvPoolLayer

plt.ioff()
DEBUG_VERBOSITY = 0
print theano.config.device

global BATCH_SIZE, EDGE_LENGTH, X_TRAIN_FPATH, Y_FPATH, X_TEST_FPATH


def inspect_inputs(i, node, fn):
    if DEBUG_VERBOSITY >= 2:
        print i, node, "input(s) value(s):", [input[0] for input in fn.inputs]


def inspect_outputs(i, node, fn):
    if DEBUG_VERBOSITY >= 2:
        print i, node, "output(s) value(s):", [output[0] for output in fn.outputs]


def compute_size(originalLen, filterLen, filterStride, poolWidth, poolStride, ignorePoolBorder=False):
    # assert poolStride == 1, 'Only supports poolStride of 1 for now. %i not accepted.' % poolStride

    print originalLen, filterLen, filterStride
    x0 = (originalLen + filterLen - 1)/filterStride

    # assumes symmetry in shape
    return downsample.DownsampleFactorMax.out_shape((x0, x0), (poolWidth, poolWidth), st=(poolStride, poolStride), ignore_border=ignorePoolBorder)[0]


def make_var(x):
    return theano.shared(x, borrow=True)


def reLU(x):
    # return x
    return T.maximum(0.0, x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


class CNN(object):

    def __init__(self, numYs,
                 numFeatureMaps,
                 imageShape,
                 filterWidths,
                 filterStrides,
                 poolWidths,
                 poolStrides,
                 n_hiddens,
                 dropOutRates,
                 initialLearningRate=0.05, learningRateDecay = 0.001,
                 L1_reg=0.00, L2_reg=0.0001,
                 rndState=0):

        """
        :param numFeatureMaps:
        :param imageShape: (image width, image height)
        :param filterWidths: [(filter width, filter height), ...] for the conv-pool layers
        :param poolWidths: (pool block width, pool block height)
        :param initialLearningRate:
        :param L1_reg:
        :param L2_reg:
        :param n_epochs:
        :param dataset:
        :param BATCH_SIZE:
        :param n_hiddens:
        :param rndState:
        :return:
        """
        assert len(numFeatureMaps) == len(filterWidths), '%i vs %i' % (len(numFeatureMaps), len(filterWidths))
        assert len(n_hiddens) == len(dropOutRates), '%i vs %i' % (len(n_hiddens), len(dropOutRates))

        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.learningRate = initialLearningRate
        self.learningRateDecay = learningRateDecay

        # TODO: make this a function
        self.config = '_'.join([str(numFeatureMaps),
                                str(imageShape),
                                str(filterWidths),
                                str(filterStrides),
                                str(poolWidths),
                                str(poolStrides),
                                str(n_hiddens),
                                str(dropOutRates)]
                               + [str(initialLearningRate), str(self.L1_reg), str(self.L2_reg)])

        print '='*5, self.config, '='*5

        rng = np.random.RandomState(rndState)

        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
        self.batchSize = BATCH_SIZE
        self.x = x
        self.y = y

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (BATCH_SIZE, nkerns[1], 4, 4)

        # ------ build conv-pool layers
        self.convPoolLayers = [None] * len(numFeatureMaps)

        # Reshape matrix of rasterized images of shape (BATCH_SIZE, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        newDim1 = x.size/(imageShape[0] * imageShape[1])       # CRAZINESS!!! GPU does NOT work with -1... REDUNKS
        layer0_input = x.reshape((newDim1, 1, imageShape[0], imageShape[1]))


        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (BATCH_SIZE, nkerns[0], 12, 12)
        self.convPoolLayers[0] = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            # image_shape=(BATCH_SIZE, 1, imageShape[0], imageShape[1]),
            filter_shape=(numFeatureMaps[0], 1, filterWidths[0], filterWidths[0]),
            poolWidth=poolWidths[0],
            poolStride=poolStrides[0],
            # activation=reLU
            # activation=lambda _x: reLU(T.tanh(_x))
            filterStride=filterStrides[0]
        )

        prevOutputImageShape = (compute_size(imageShape[0], filterWidths[0], filterStrides[0], poolWidths[0], poolStrides[0]),
                                compute_size(imageShape[1], filterWidths[0], filterStrides[0], poolWidths[0], poolStrides[0]))
        print 'prevOutputImageShape', prevOutputImageShape

        for i in range(1, len(self.convPoolLayers)):

            self.convPoolLayers[i] = LeNetConvPoolLayer(
                rng,
                input=self.convPoolLayers[i-1].output,
                image_shape=(BATCH_SIZE, numFeatureMaps[i-1], prevOutputImageShape[0], prevOutputImageShape[1]),
                filter_shape=(numFeatureMaps[i], numFeatureMaps[i-1], filterWidths[i], filterWidths[i]),
                poolWidth=poolWidths[i],
                poolStride=poolStrides[i],
                filterStride=filterStrides[i]
                # activation=lambda x: reLU(T.tanh(x))
                # activation=reLU
            )

            # prevOutputImageShape = ((prevOutputImageShape[0] - filterShapes[i][0] + 1)/poolWidths[i],
            #                         (prevOutputImageShape[1] - filterShapes[i][1] + 1)/poolWidths[i])
            prevOutputImageShape = (compute_size(prevOutputImageShape[0], filterWidths[i], filterStrides[i], poolWidths[i], poolStrides[i]),
                                    compute_size(prevOutputImageShape[1], filterWidths[i], filterStrides[i], poolWidths[i], poolStrides[i]))
            print 'prevOutputImageShape', prevOutputImageShape


        # ------ build hidden layers

        self.fullyConnectedLayers = [None] * len(n_hiddens)

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (BATCH_SIZE, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (BATCH_SIZE, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        fcLayer_input = self.convPoolLayers[-1].output.flatten(2)

        self.fullyConnectedLayers[0] = HiddenLayer(
            rng,
            input=fcLayer_input,
            n_in=numFeatureMaps[-1] * prevOutputImageShape[0] * prevOutputImageShape[1],
            n_out=n_hiddens[0],
            random_seed=None,
            # activation=T.tanh,
            # activation=lambda x: reLU(T.tanh(x)),
            # activation=reLU,
            drop_out_rate=dropOutRates[0]
        )

        for i in range(1, len(self.fullyConnectedLayers)):
            self.fullyConnectedLayers[i] = HiddenLayer(
                rng,
                input=self.fullyConnectedLayers[i-1].output,
                n_in=n_hiddens[i-1],
                n_out=n_hiddens[i],
                random_seed=None,
                # activation=lambda x: reLU(T.tanh(x)),
                # activation=reLU,
                drop_out_rate=dropOutRates[i]
            )


        # ------ build the last layer -- Logistic Regression layer

        # classify the values of the fully-connected sigmoidal layer
        self.lastLayer = LogisticRegression(input=self.fullyConnectedLayers[-1].output, n_in=n_hiddens[-1], n_out=numYs)

        self.layers = self.convPoolLayers + self.fullyConnectedLayers + [self.lastLayer]


        ######################
        #        TRAIN       #
        ######################

        # training parameters
        index = T.scalar(dtype='int32')  # index to a [mini]batch

        # TODO: move this to training time
        (self.train_set_x, self.train_set_y), (self.valid_set_x, self.valid_set_y), (self.test_set_x, self.test_set_y) = read_train_data(X_TRAIN_FPATH, Y_FPATH)

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.lastLayer.params \
                      + [item for l in self.fullyConnectedLayers for item in l.params] \
                      + [item for l in self.convPoolLayers for item in l.params]

        # the cost we minimize during training is the NLL of the model
        self.cost = self.lastLayer.negative_log_likelihood(y) \
                    + L2_reg * sum(l.L2 for l in self.layers)


        # create a list of gradients for all model parameters
        self.grads = T.grad(self.cost, self.params)

        self.updates = [
            (param_i, (param_i - self.learningRate * grad_i).astype(theano.config.floatX))
            for param_i, grad_i in zip(self.params, self.grads)
        ]


        self.train_model = theano.function(
            [index],
            self.cost,
            updates=self.updates,
            givens={
                x: self.train_set_x[index * self.batchSize: (index + 1) * self.batchSize],
                y: self.train_set_y[index * self.batchSize: (index + 1) * self.batchSize]
            },
            name='train_model',
            # mode=theano.compile.MonitorMode(
            #     pre_func=inspect_inputs,
            #     post_func=inspect_outputs),
            # mode='FAST_COMPILE',
            allow_input_downcast=True
        )

        # create a function to compute the mistakes that are made by the model
        self.test_model = theano.function(
            [index],
            self.lastLayer.negative_log_likelihood(y),
            givens={
                x: self.test_set_x[index * BATCH_SIZE: (index + 1) * BATCH_SIZE],
                y: self.test_set_y[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]
            },
            name='test_model',
            mode=theano.compile.MonitorMode(
                pre_func=inspect_inputs,
                post_func=inspect_outputs),
            allow_input_downcast=True
        )

        self.validate_model = theano.function(
            [index],
            self.lastLayer.negative_log_likelihood(y),
            givens={
                x: self.valid_set_x[index * BATCH_SIZE: (index + 1) * BATCH_SIZE],
                y: self.valid_set_y[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]
            },
            name='validation_model',
            mode=theano.compile.MonitorMode(
                pre_func=inspect_inputs,
                post_func=inspect_outputs),
            allow_input_downcast=True
        )

        self.predict_model = theano.function(
            [x],
            self.lastLayer.p_y_given_x,
            name='predict_model',
            mode=theano.compile.MonitorMode(
                pre_func=inspect_inputs,
                post_func=inspect_outputs)
        )

        self.print_stuff = theano.function(
            [index],
            [

            ],

            # + theano.printing.Print('x shape', attrs=['shape'])(x)],
            # + [l.output_print for l in self.convPoolLayers]
            # + [self.fullyConnectedLayer.output_print,
            #    self.lastLayer.t_dot_print,
            #    self.lastLayer.p_y_given_x_print],

            givens={
                x: self.train_set_x[index * BATCH_SIZE: (index + 1) * BATCH_SIZE],
                y: self.train_set_y[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]
            },
            on_unused_input='ignore'
        ) if DEBUG_VERBOSITY > 0 else None

        print 'Done building CNN object.'

    def train(self, saveParameters, n_epochs, patience):

        # compute number of minibatches for training, validation and testing
        # NOTE: skipping the last mini-batch on purpose in case of imperfect division for GPU speed
        n_train_batches = self.train_set_x.get_value(borrow=True).shape[0] / self.batchSize
        n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0] / self.batchSize
        n_test_batches = self.test_set_x.get_value(borrow=True).shape[0] / self.batchSize

        train(self, saveParameters, n_train_batches, n_valid_batches, n_test_batches,
              self.train_model, self.validate_model, self.test_model, self.print_stuff,
              n_epochs, patience=patience)

    def saveParams(self, suffix=''):
        print 'Saving parameters...'
        f = file('/Users/jennyyuejin/K/tryTheano/params_%s%s.save' %
                 (self.config,
                  '' if suffix=='' else '_' + str(suffix)),
                 'wb')
        cPickle.dump(self.params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def predict(self, outputDir, chunksize=5000, takeNumColumns=None):

        outputFpath = os.path.join(outputDir,
                                   '%s_%s.csv' %
                                   (datetime.date.today().strftime('%b%d%Y'), self.config)
        )
        outputFile = file(outputFpath, 'w')

        # write headers
        headers = ['image'] + CLASS_NAMES
        outputFile.write(','.join(headers))
        outputFile.write('\n')

        # write contents
        reader = pandas.read_csv(X_TEST_FPATH, chunksize = chunksize, header=None)

        print '========= Saving prediction results for %s to %s.' % (X_TEST_FPATH, outputFpath)

        for i, chunk in enumerate(reader):

            print 'chunk', i

            pred_x, testFnames = read_test_data_in_chunk(chunk, takeNumColumns=takeNumColumns)

            pred_results = self.predict_model(pred_x)

            pandas.DataFrame(pred_results, index=testFnames).reset_index().to_csv(outputFile, header=False, index=False)

            outputFile.flush()

        outputFile.close()

        return outputFpath

    def calculate_last_layer_train(self, fname):

        numOutPoints = self.fullyConnectedLayers[-1].n_out
        fullx = np.array(pandas.read_csv(fname, header=None, dtype=theano.config.floatX))
        numTrain = fullx.shape[0]
        res = np.empty((numTrain, numOutPoints))

        for i in range(numTrain/BATCH_SIZE + 1):
            print i
            res[i*BATCH_SIZE : (i+1)*BATCH_SIZE, :] = self.fullyConnectedLayers[-1].output.eval({self.x: fullx[i*BATCH_SIZE : (i+1)*BATCH_SIZE]})

        return res

    def calculate_last_layer_test(self, inputFname, chunksize=1000):

        numOutPoints = self.fullyConnectedLayers[-1].n_out

        res = np.empty((0, numOutPoints))
        reader = pandas.read_csv(inputFname, chunksize = chunksize, header=None)

        for i, chunk in enumerate(reader):

            print 'chunk', i
            pred_x, _ = read_test_data_in_chunk(chunk)
            vals = self.fullyConnectedLayers[-1].output.eval({self.x: pred_x})

            res = np.append(res, vals, axis=0)

        return res

    def fill_in_parameters(self, fpath):

        params = cPickle.load(file(fpath, 'rb'))

        assert len(params) == 2 + 2 * len(self.fullyConnectedLayers) + 2 * len(self.convPoolLayers), \
            '%i vs %i + 2 * %i + 2 * %i' % (len(params), 2, len(self.fullyConnectedLayers), len(self.convPoolLayers))

        self.lastLayer.W.set_value(params[0].get_value())
        self.lastLayer.b.set_value(params[1].get_value())

        numSkip = 2
        for i, fcLayer in enumerate(self.fullyConnectedLayers):
            fcLayer.W.set_value(params[numSkip + 2*i].get_value())
            fcLayer.b.set_value(params[numSkip + 2*i + 1].get_value())
        numSkip += 2*len(self.fullyConnectedLayers)

        for i, cpLayer in enumerate(self.convPoolLayers):
            cpLayer.W.set_value(params[numSkip + 2*i].get_value())
            cpLayer.b.set_value(params[numSkip + 2*i + 1].get_value())

    def shuffle_training_data(self, random_seed=None):

        print 'Shuffling training X and y data...'

        numRows = self.train_set_x.shape[0]

        srng = RandomStreams(seed = random_seed)
        mask = srng.permutation(n=numRows, size=(1,)).reshape((numRows,))

        self.train_set_x = self.train_set_x[mask]
        self.train_set_y = self.train_set_y[mask]


    @classmethod
    def create_class_obj_from_file(cls, fpath):

        # parse parameters from the filename
        containingDir, fname = os.path.split(fpath)     # strip directory
        fname, _ = os.path.splitext(fname)              # strip extension
        fname = fname.replace('(', '[').replace(')',']').replace('None', 'null')

        # extract variables
        numFeatureMaps, imageShape, filterShapes, poolWidths, n_hiddens, dropOutRates, learningRate, L1, L2 \
            = [json.loads(s) for s in fname.split('_')[1:10]]

        print 'Loading variables:'
        pprint(dict(zip(['numFeatureMaps', 'imageShape', 'filterShapes', 'poolWidths', 'n_hiddens', 'dropOutRates', 'learningRate', 'L1', 'L2'],
                        [numFeatureMaps, imageShape, filterShapes, poolWidths, n_hiddens, dropOutRates, learningRate, L1, L2])))

        # initialize the CNN object
        obj = CNN(len(CLASS_NAMES), numFeatureMaps, imageShape, filterShapes, poolWidths,
                  n_hiddens, dropOutRates,
                  initialLearningRate=learningRate, L1_reg=L1, L2_reg=L2)

        # fill in parameters
        obj.fill_in_parameters(fpath)

        return obj


def train(cnnObj, saveParameters, n_train_batches, n_valid_batches, n_test_batches,
          train_model, validate_model, test_model, print_stuff,
          n_epochs, patience = 2000, patience_increase = 2, improvement_threshold = 0.999):

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    # patience = 10000  # look as this many examples regardless
    # patience_increase = 2  # wait this much longer when a new best is
    # found
    # improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = n_train_batches/4
    shuffling_frequency = 2     # shuffle the training data every this many epochs
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch
    miniBatchPrints = range(n_train_batches)[::10]

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    for epoch in range(n_epochs):
        print '----- epoch:', epoch, patience

        if epoch % shuffling_frequency == 0:
            cnnObj.shuffle_training_data()

        cnnObj.learningRate /= 1 + cnnObj.learningRateDecay * cnnObj.learningRate * epoch
        print 'Updating learning rate to be', cnnObj.learningRate

        print 'minibatch:'
        for minibatch_index in xrange(n_train_batches):

            if DEBUG_VERBOSITY > 0:
                print_stuff(minibatch_index)

            curTrainCost = train_model(minibatch_index)

            if minibatch_index in miniBatchPrints:
                print minibatch_index, 'minibatch_avg_cost =', curTrainCost

            # iteration number
            iter = epoch * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, val error %f (vs best val error of %f)' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss,
                        best_validation_loss
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience += patience_increase

                    if saveParameters:
                        cnnObj.saveParams(suffix=BATCH_SIZE)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    # test_losses = [test_model(i) for i
                    #                in xrange(n_test_batches)]
                    # test_score = np.mean(test_losses)
                    #
                    # print(('     epoch %i, minibatch %i/%i, test error of '
                    #        'best model %f %%') %
                    #       (epoch, minibatch_index + 1, n_train_batches,
                    #        test_score * 100.))
                # elif this_validation_loss > best_validation_loss*1.05:   # shooting over in the wrong direction
                #     print 'DECREASING rate-decreasing-multiple from %f to %f.' % (rate_dec_multiple, rate_dec_multiple/1.15)
                #     rate_dec_multiple /= 1.15
                else:
                    cnnObj.learningRateDecay *= 1.1
                    print 'Bumping learning rate decay to', cnnObj.learningRateDecay

        if epoch > patience:
            'OUT OF PATIENCE. EXITING...'
            break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    try:
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
    except:
        pass



def read_train_data(xtrainfpath,
                    yfpath = '/Users/jennyyuejin/K/NDSB/Data/y.csv'):

    # read training data
    print xtrainfpath
    x_data = np.array(pandas.read_csv(xtrainfpath, header=None), dtype=theano.config.floatX)[:, : (EDGE_LENGTH**2)]
    y_data = np.array(pandas.read_csv(yfpath, header=None), dtype=theano.config.floatX).ravel()

    print x_data.dtype

    # TODO: should ideally should StratifiedShuffleSplit, which doesn't work with tiny data sets when some class has only one data point.
    temp, testInds = ShuffleSplit(len(y_data), n_iter=1, test_size=0.1, random_state=1)._iter_indices().next()
    testX, testY = x_data[testInds], y_data[testInds]

    trainInds, validInds = ShuffleSplit(len(y_data[temp]), n_iter=1, test_size=0.1, random_state=1)._iter_indices().next()
    trainX, trainY = x_data[temp][trainInds], y_data[temp][trainInds]
    validX, validY = x_data[temp][validInds], y_data[temp][validInds]

    print trainX.dtype, validX.dtype, testX.dtype, testY.dtype
    print trainX.shape, validX.shape, testX.shape

    return [(make_var(trainX), T.cast(make_var(trainY), 'int32')),
            (make_var(validX), T.cast(make_var(validY), 'int32')),
            (make_var(testX), T.cast(make_var(testY), 'int32'))]


def read_test_data_in_chunk(chunk):
    """
    :param chunk: pandas DataFrame
    :return:
    """

    chunk = np.array(chunk)

    testFnames = chunk[:, 0]
    testData = np.array(chunk[:, 1:], dtype=theano.config.floatX)[:, :(EDGE_LENGTH**2)]

    return testData, testFnames


if __name__ == '__main__':

    BATCH_SIZE = 30
    EDGE_LENGTH = 48

    # X_TRAIN_FPATH = '/Users/jennyyuejin/K/NDSB/Data/X_train_%i_%i_simple.csv' % (EDGE_LENGTH, EDGE_LENGTH)
    X_TRAIN_FPATH = '/Users/jennyyuejin/K/NDSB/Data/tinyX.csv'
    # X_TEST_FPATH = '/Users/jennyyuejin/K/NDSB/Data/X_test_%i_%i_simple.csv' % (EDGE_LENGTH, EDGE_LENGTH)
    # Y_FPATH = '/Users/jennyyuejin/K/NDSB/Data/y.csv'
    Y_FPATH = '/Users/jennyyuejin/K/NDSB/Data/tinyY.csv'



    # ------ train
    # cnnObj = CNN(len(CLASS_NAMES),
    #              numFeatureMaps = [4, 4, 4, 3, 3, 3, 3],
    #              imageShape = [EDGE_LENGTH, EDGE_LENGTH],
    #              filterShapes = [(4, 4), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
    #              poolWidths = [2, 1, 1, 1, 1, 1, 1],
    #              n_hiddens=[512, 200],
    #              dropOutRates=[0.5, 0.2],
    #              initialLearningRate=0.1,
    #              L1_reg=0, L2_reg=0.00001,
    #              )

    # cnnObj = CNN(len(CLASS_NAMES),
    #              numFeatureMaps = [4, 3],
    #              imageShape = [EDGE_LENGTH, EDGE_LENGTH],
    #              filterShapes = [(4, 4), (2, 2)],
    #              poolWidths = [2, 2],
    #              n_hiddens=[512, 200],
    #              dropOutRates=[0.5, 0.2],
    #              initialLearningRate=0.05,
    #              L1_reg=0, L2_reg=0.01,
    #              )

    cnnObj = CNN(len(CLASS_NAMES),
                 numFeatureMaps = [96, 128, 128],
                 imageShape = [EDGE_LENGTH, EDGE_LENGTH],
                 filterWidths= [5, 3, 3],
                 filterStrides = [4, 1, 1],
                 poolWidths = [3, 1, 3],
                 poolStrides = [2, 1, 2],
                 # poolStrides=[3, 1, 3],
                 n_hiddens=[512, 512],
                 dropOutRates=[0.5, 0.5],
                 initialLearningRate=0.1,
                 L1_reg=0, L2_reg=0.001,
                 )
    # cnnObj.fill_in_parameters('/Users/jennyyuejin/K/tryTheano/params_[96, 128, 128]_[48, 48]_[(5, 5), (3, 3), (3, 3)]_[3, 1, 3]_[512, 512]_[0.5, 0.5]_0.1_0_0.001_40.save')

    numEpochs = 400
    cnnObj.train(saveParameters = False, n_epochs=numEpochs, patience=20000)


    # ------ predict
    # cnnObj.predict('/Users/jennyyuejin/K/NDSB/Data/submissions', chunksize=5000)

    #
    # fpath = '/Users/jennyyuejin/K/tryTheano/params_[96, 128, 128]_[48, 48]_[(5, 5), (3, 3), (3, 3)]_[3, 1, 3]_[512, 512]_[0.5, 0.5]_0.1_0_0.001_25.save'
    # obj = CNN.create_class_obj_from_file(fpath)
    # res_train = obj.calculate_last_layer_train('/Users/jennyyuejin/K/NDSB/Data/X_train_48_48_simple.csv')
    # res_test = obj.calculate_last_layer_test('/Users/jennyyuejin/K/NDSB/Data/X_test_48_48_simple.csv', 10000)


    # params = cPickle.load(file(fpath, 'rb'))
    # pprint(params)
    #
    #
    # def calculate_last_train_layer():
    #
    #     numOutPoints = 200
    #     fullx = np.array(pandas.read_csv('/Users/jennyyuejin/K/NDSB/Data/X_train_48_48_simple.csv', header=None, dtype=theano.config.floatX))
    #     numTrain = fullx.shape[0]
    #     res = np.empty((numTrain, numOutPoints))
    #
    #     for i in range(numTrain/BATCH_SIZE + 1):
    #         print i
    #         res[i*BATCH_SIZE : (i+1)*BATCH_SIZE, :] = obj.fullyConnectedLayers[-1].output.eval({obj.x: fullx[i*BATCH_SIZE : (i+1)*BATCH_SIZE]})
    #
    #     np.save('/Users/jennyyuejin/K/NDSB/Data/lastlayeroutput.save', res)
    #
    #
    #     outputFile = file('/Users/jennyyuejin/K/NDSB/Data/lastlayerout_test.csv', 'a')
    #     reader = pandas.read_csv(X_TEST_FPATH, chunksize = 1000, header=None)
    #
    #     for i, chunk in enumerate(reader):
    #
    #         print 'chunk', i
    #         pred_x, testFnames = read_test_data_in_chunk(chunk)
    #         vals = obj.fullyConnectedLayers[-1].output.eval({obj.x: pred_x})
    #         np.savetxt(outputFile, vals)
    #         outputFile.flush()
    #
    #     outputFile.close()


