__author__ = 'JennyYueJin'


import cPickle
import os
import sys
import time
import datetime
import numpy as np
import pandas
from matplotlib import pyplot as plt
from PIL import Image
from pprint import pprint

from sklearn.cross_validation import StratifiedShuffleSplit

import theano
import theano.tensor as T
theano.config.openmp = True
theano.config.optimizer = 'None'
theano.config.exception_verbosity='high'

sys.path.extend(['/Users/jennyyuejin/K/tryTheano'])

from logisticRegressionExample import LogisticRegression, load_data
from mlpExample import HiddenLayer

from NDSB.fileMangling import make_submission_file
from NDSB.global_vars import CLASS_NAMES
from leNetConvPoolLayer import LeNetConvPoolLayer

plt.ioff()
DEBUG_VERBOSITY = 0
print theano.config.device

global X_TRAIN_FPATH, Y_FPATH, X_TEST_FPATH


def inspect_inputs(i, node, fn):
    if DEBUG_VERBOSITY >= 2:
        print i, node, "input(s) value(s):", [input[0] for input in fn.inputs]


def inspect_outputs(i, node, fn):
    if DEBUG_VERBOSITY >= 2:
        print i, node, "output(s) value(s):", [output[0] for output in fn.outputs]


def make_var(x):
    return theano.shared(x, borrow=True)


class CNN(object):

    def __init__(self, numYs,
                 numFeatureMaps,
                 imageShape,
                 filterShapes,
                 poolWidths,
                 initialLearningRate=0.01, L1_reg=0.00, L2_reg=0.0001, batch_size=100, n_hidden=100,
                 rndState=0):

        """
        :param numFeatureMaps:
        :param imageShape: (image width, image height)
        :param filterShapes: [(filter width, filter height), ...] for the conv-pool layers
        :param poolWidths: (pool block width, pool block height)
        :param initialLearningRate:
        :param L1_reg:
        :param L2_reg:
        :param n_epochs:
        :param dataset:
        :param batch_size:
        :param n_hidden:
        :param rndState:
        :return:
        """
        assert len(numFeatureMaps) == len(filterShapes)

        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        self.config = '_'.join([str(numFeatureMaps)]
                               + [str(i) for i in imageShape]
                               + [str(f[0]) for f in filterShapes]
                               + [str(i) for i in poolWidths] +
                               [str(n_hidden), str(initialLearningRate), str(self.L1_reg), str(self.L2_reg)])
        print '='*5, self.config, '='*5

        rng = np.random.RandomState(rndState)

        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
        self.batchSize = batch_size

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)

        # ------ build conv-pool layers
        self.convPoolLayers = [None] * len(numFeatureMaps)

        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        layer0_input = x.reshape((-1, 1, imageShape[0], imageShape[1]))
        prevOutputImageShape = ((imageShape[0] - filterShapes[0][0] + 1)/poolWidths[0],
                                (imageShape[1] - filterShapes[0][1] + 1)/poolWidths[0])

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        self.convPoolLayers[0] = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 1, imageShape[0], imageShape[1]),
            filter_shape=(numFeatureMaps[0], 1, filterShapes[0][0], filterShapes[0][1]),
            poolsize=[poolWidths[0], poolWidths[0]]
        )


        for i in range(1, len(self.convPoolLayers)):

            self.convPoolLayers[i] = LeNetConvPoolLayer(
                rng,
                input=self.convPoolLayers[i-1].output,
                image_shape=(batch_size, numFeatureMaps[i-1], prevOutputImageShape[0], prevOutputImageShape[1]),
                filter_shape=(numFeatureMaps[i], numFeatureMaps[i-1], filterShapes[i][0], filterShapes[i][1]),
                poolsize=[poolWidths[i], poolWidths[i]]
            )

            prevOutputImageShape = ((prevOutputImageShape[0] - filterShapes[i][0] + 1)/poolWidths[i],
                                    (prevOutputImageShape[1] - filterShapes[i][1] + 1)/poolWidths[i])


        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        fcLayer_input = self.convPoolLayers[-1].output.flatten(2)

        # construct a fully-connected sigmoidal layer
        self.fullyConnectedLayer = HiddenLayer(
            rng,
            input=fcLayer_input,
            n_in=numFeatureMaps[-1] * prevOutputImageShape[0] * prevOutputImageShape[1],
            n_out=n_hidden,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        self.lastLayer = LogisticRegression(input=self.fullyConnectedLayer.output, n_in=n_hidden, n_out=numYs)

        self.layers = self.convPoolLayers + [self.lastLayer, self.fullyConnectedLayer]


        ######################
        #        TRAIN       #
        ######################

        # training parameters
        index = T.lscalar()  # index to a [mini]batch
        rate_dec_multiple = T.fscalar()         # index to epoch, used for decreasing the learning rate over time
        rate_dec_multiple_given = T.fscalar()   # index to epoch, used for decreasing the learning rate over time

        # TODO: move this to training time
        (self.train_set_x, self.train_set_y), (self.valid_set_x, self.valid_set_y), (self.test_set_x, self.test_set_y) = read_train_data(X_TRAIN_FPATH, Y_FPATH)

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.lastLayer.params \
                      + self.fullyConnectedLayer.params \
                      + [item for l in self.convPoolLayers for item in l.params]

        # the cost we minimize during training is the NLL of the model
        self.cost = self.lastLayer.negative_log_likelihood(y) \
                    + L2_reg * sum(l.L2 for l in self.layers)


        # create a list of gradients for all model parameters
        self.grads = T.grad(self.cost, self.params)

        self.updates = [
            (param_i, (param_i - initialLearningRate / (1 + 0.005 * rate_dec_multiple) * grad_i).astype(theano.config.floatX))
            for param_i, grad_i in zip(self.params, self.grads)
        ]


        self.train_model = theano.function(
            [index, rate_dec_multiple_given],
            self.cost,
            updates=self.updates,
            givens={
                x: self.train_set_x[index * self.batchSize: (index + 1) * self.batchSize],
                y: self.train_set_y[index * self.batchSize: (index + 1) * self.batchSize],
                rate_dec_multiple: rate_dec_multiple_given
            },
            name='train_model',
            mode=theano.compile.MonitorMode(
                pre_func=inspect_inputs,
                post_func=inspect_outputs),
            allow_input_downcast=True
        )

        # create a function to compute the mistakes that are made by the model
        self.test_model = theano.function(
            [index],
            self.lastLayer.negative_log_likelihood(y),
            givens={
                x: self.test_set_x[index * batch_size: (index + 1) * batch_size],
                y: self.test_set_y[index * batch_size: (index + 1) * batch_size]
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
                x: self.valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: self.valid_set_y[index * batch_size: (index + 1) * batch_size]
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
                theano.printing.Print('last', attrs=['__str__'])(self.lastLayer.L1),
                theano.printing.Print('hidden', attrs=['__str__'])(self.fullyConnectedLayer.L1),
                theano.printing.Print('convPools', attrs=['__str__'])(self.convPoolLayers[0].L1)
            ],
            # + theano.printing.Print('x shape', attrs=['shape'])(x)],
            # + [l.output_print for l in self.convPoolLayers]
            # + [self.fullyConnectedLayer.output_print,
            #    self.lastLayer.t_dot_print,
            #    self.lastLayer.p_y_given_x_print],

            givens={
                x: self.train_set_x[index * batch_size: (index + 1) * batch_size],
                y: self.train_set_y[index * batch_size: (index + 1) * batch_size]
            },
            on_unused_input='warn'
        ) if DEBUG_VERBOSITY > 0 else None

        print 'Done building CNN object.'

    def train(self, saveParameters, n_epochs=1000, patience=1000):

        # compute number of minibatches for training, validation and testing
        n_train_batches = self.train_set_x.get_value(borrow=True).shape[0] / self.batchSize + 1
        n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0] / self.batchSize + 1
        n_test_batches = self.test_set_x.get_value(borrow=True).shape[0] / self.batchSize + 1

        train(n_train_batches, n_valid_batches, n_test_batches,
              self.train_model, self.validate_model, self.test_model, self.print_stuff,
              n_epochs, patience=patience)

        if saveParameters:
            print 'Saving parameters...'
            f = file('/Users/jennyyuejin/K/tryTheano/params_%s.save' % self.config, 'wb')
            cPickle.dump(self.params, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

    def predict(self, outputDir, chunksize=10000, takeNumColumns=None):

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

        print '========= Saving prediction results for %s to %s.' (X_TEST_FPATH, outputFpath)

        for i, chunk in enumerate(reader):

            print 'chunk', i

            pred_x, testFnames = read_test_data_in_chunk(chunk, takeNumColumns=takeNumColumns)

            pred_results = self.predict_model(pred_x)

            pandas.DataFrame(pred_results, index=testFnames).reset_index().to_csv(outputFile, header=False)

            outputFile.flush()

        outputFile.close()

        return outputFpath


def train(n_train_batches, n_valid_batches, n_test_batches,
          train_model, validate_model, test_model, print_stuff,
          n_epochs, patience = 2000, patience_increase = 2, improvement_threshold = 0.995):

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
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    rate_dec_multiple = epoch
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        print '----- epoch:', epoch, patience

        print 'minibatch:'
        for minibatch_index in xrange(n_train_batches):

            print minibatch_index,

            if DEBUG_VERBOSITY:
                print_stuff(minibatch_index)

            print 'minibatch_avg_cost =', train_model(minibatch_index, rate_dec_multiple)

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

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
                        # patience = max(patience, iter * patience_increase)
                        patience += patience_increase * n_train_batches

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                else:   # we are going too fast
                    print 'Bumping rate-decreasing-multiple from %f to %f.' % (rate_dec_multiple, rate_dec_multiple*1.1)
                    rate_dec_multiple *= 1.1

            if patience <= iter:
                done_looping = True
                break

        epoch += 1
        rate_dec_multiple += 1

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
                    yfpath = '/Users/jennyyuejin/K/NDSB/Data/y.csv',
                    takeNumColumns = None):

    # read training data
    x_data = np.array(pandas.read_csv(xtrainfpath, header=None), dtype=theano.config.floatX)
    y_data = np.array(pandas.read_csv(yfpath, header=None), dtype=theano.config.floatX).ravel()

    numRows = x_data.shape[0]

    x_data = x_data[:numRows, :]
    y_data = y_data[:numRows]

    if takeNumColumns is not None:
        x_data = x_data[:, :takeNumColumns]

    temp, testInds = StratifiedShuffleSplit(y_data, n_iter=1, test_size=0.1, random_state=1)._iter_indices().next()
    testX, testY = x_data[testInds], y_data[testInds]

    trainInds, validInds = StratifiedShuffleSplit(y_data[temp], n_iter=1, test_size=0.1, random_state=1)._iter_indices().next()
    trainX, trainY = x_data[temp][trainInds], y_data[temp][trainInds]
    validX, validY = x_data[temp][validInds], y_data[temp][validInds]

    return [(make_var(trainX), T.cast(make_var(trainY), 'int32')),
            (make_var(validX), T.cast(make_var(validY), 'int32')),
            (make_var(testX), T.cast(make_var(testY), 'int32'))]


def read_test_data_in_chunk(chunk, takeNumColumns=None):
    """
    :param chunk: pandas DataFrame
    :return:
    """

    chunk = np.array(chunk)

    testFnames = chunk[:, 0]
    testData = np.array(chunk[:, 1:], dtype=theano.config.floatX)

    if takeNumColumns is not None:
        testData = testData[:, :takeNumColumns]

    return testData, testFnames

# def read_data(xtrainfpath,
#               xtestfpath = '/Users/jennyyuejin/K/NDSB/Data/X_test_15_15.csv',
#               yfpath = '/Users/jennyyuejin/K/NDSB/Data/y.csv',
#               takeNumColumns = None):
#
#     return read_train_data(xtrainfpath, yfpath, takeNumColumns=takeNumColumns), \
#            read_test_data(xtestfpath, takeNumColumns=takeNumColumns)


if __name__ == '__main__':


    # open random image of dimensions 272 x 328
    # img = Image.open(open('/Users/jennyyuejin/K/tryTheano/Data/four.png'))
    # img_shape = (25, 25)
    #
    # # dimensions are (height, width, channel)
    # img = np.array(img.getdata(), dtype='float64') / 256.
    # img = img[: img_shape[0]*img_shape[1], 0].reshape(1, img_shape[0] * img_shape[1])

    batchSize = 250
    edgeLength = 48

    # trainData, testData, testFnames = read_data(xtrainfpath= '/Users/jennyyuejin/K/NDSB/Data/X_train_%i_%i_simple.csv' % (edgeLength, edgeLength),
    #                                             xtestfpath = '/Users/jennyyuejin/K/NDSB/Data/X_test_%i_%i_simple.csv' % (edgeLength, edgeLength),
    #                                             takeNumColumns=edgeLength*edgeLength)

    X_TRAIN_FPATH= '/Users/jennyyuejin/K/NDSB/Data/X_train_%i_%i_simple.csv' % (edgeLength, edgeLength)
    X_TEST_FPATH = '/Users/jennyyuejin/K/NDSB/Data/X_test_%i_%i_simple.csv' % (edgeLength, edgeLength)
    Y_FPATH = '/Users/jennyyuejin/K/NDSB/Data/y.csv'

    cnnObj = CNN(121,
                 numFeatureMaps = [4, 4, 4, 3, 3, 3, 3],
                 imageShape = [edgeLength, edgeLength],
                 filterShapes = [(4, 4), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
                 poolWidths = [1, 2, 1, 1, 1, 1, 1],
                 initialLearningRate=0.05, batch_size=batchSize, n_hidden=350,
                 L1_reg=0, L2_reg=0.0003,
                 )

    cnnObj.train(saveParameters=False, n_epochs=1, patience=100)

    cnnObj.predict('/Users/jennyyuejin/K/NDSB/Data/submissions', chunksize=5000)



        # plot original image and first and second components of output
        # plt.subplot(1, 3, 1); plt.axis('off'); plt.imshow(img)
        # plt.gray()
        #
        # # recall that the convOp output (filtered image) is actually a "minibatch",
        # # of size 1 here, so we take index 0 in the first dimension:
        # plt.subplot(1, 3, 2); plt.axis('off'); plt.imshow(filtered_img[0, 0, :, :])
        # plt.subplot(1, 3, 3); plt.axis('off'); plt.imshow(filtered_img[0, 1, :, :])
        #
        # plt.show()
