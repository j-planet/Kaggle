__author__ = 'JennyYueJin'


import cPickle
import gzip
import os
import sys
import time
import numpy as np
import pandas
from matplotlib import pyplot as plt
from PIL import Image
from pprint import pprint

from sklearn.cross_validation import StratifiedShuffleSplit

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
theano.config.openmp = True
theano.config.optimizer = 'None'
theano.config.exception_verbosity='high'

sys.path.extend(['/Users/jennyyuejin/K/tryTheano'])

from logisticRegressionExample import LogisticRegression, load_data
from mlpExample import HiddenLayer

from NDSB.fileMangling import make_submission_file

plt.ioff()
debugMode = False

def inspect_inputs(i, node, fn):
    if debugMode:
        print i, node, "input(s) value(s):", [input[0] for input in fn.inputs]


def inspect_outputs(i, node, fn):
    if debugMode:
        print i, node, "output(s) value(s):", [output[0] for output in fn.outputs]


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width" inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))

        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))

        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            # filter_shape=filter_shape,
            # image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_print = theano.printing.Print('LeNet output', attrs=['__str__', 'shape'])(self.output)

        # store parameters of this layer
        self.params = [self.W, self.b]


def make_var(x):
    return theano.shared(x, borrow=True)


def run_cnn(datasets,
            numYs,
            numFeatureMaps,
            imageShape,
            filterShape,
            poolSize,
            learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
            dataset='/Users/JennyYueJin/K/tryTheano/Data/mnist.pkl.gz',
            batch_size=100,
            n_hidden=100, rndState=0,
            predict_set_x=None, testFnames=None):

    """
    :param numFeatureMaps:
    :param imageShape: (image width, image height)
    :param filterShape: (filter width, filter height)
    :param poolSize: (pool block width, pool block height)
    :param learning_rate:
    :param L1_reg:
    :param L2_reg:
    :param n_epochs:
    :param dataset:
    :param batch_size:
    :param n_hidden:
    :param rndState:
    :return:
    """

    # datasets = load_data(dataset)

    rng = np.random.RandomState(rndState)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size + 1
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size + 1
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size + 1

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((-1, 1, imageShape[0], imageShape[1]))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, imageShape[0], imageShape[1]),
        filter_shape=(numFeatureMaps[0], 1, filterShape[0], filterShape[1]),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1_imgWidth = (imageShape[0]-filterShape[0]+1)/poolSize[0]
    layer1_imgHeight = (imageShape[1]-filterShape[1]+1)/poolSize[1]

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, numFeatureMaps[0], layer1_imgWidth, layer1_imgHeight),
        filter_shape=(numFeatureMaps[1], numFeatureMaps[0], filterShape[0], filterShape[1]),
        poolsize=poolSize
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)
    layer2_imgWidth = (layer1_imgWidth-filterShape[0]+1)/poolSize[0]
    layer2_imgHeight = (layer1_imgHeight-filterShape[1]+1)/poolSize[1]

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=numFeatureMaps[1] * layer2_imgWidth * layer2_imgHeight,
        n_out=n_hidden,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=n_hidden, n_out=numYs)

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)
    # + L1_reg * abs(params).sum() + L2_reg * (params**2).sum()

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        },
        name='test_model',
        mode=theano.compile.MonitorMode(
            pre_func=inspect_inputs,
            post_func=inspect_outputs)
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        },
        name='validation_model',
        mode=theano.compile.MonitorMode(
            pre_func=inspect_inputs,
            post_func=inspect_outputs)
    )

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    print_stuff = theano.function(
        [index],
        [theano.printing.Print('x shape', attrs=['shape'])(x),
         layer0.output_print, layer1.output_print, layer2.output_print,
         layer3.t_dot_print, layer3.p_y_given_x_print],
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='warn'
    ) if debugMode else None

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        name='train_model',
        mode=theano.compile.MonitorMode(
            pre_func=inspect_inputs,
            post_func=inspect_outputs)
    )

    train(n_train_batches, n_valid_batches, n_test_batches,
          train_model, validate_model, test_model, print_stuff,
          n_epochs)

    f = file('/Users/jennyyuejin/K/tryTheano/params_%i.save' % imageShape[0], 'wb')
    cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    # predict
    if predict_set_x is not None:

        predict_model = theano.function(
            [],
              layer3.p_y_given_x,
              givens={
                  x: predict_set_x
              },
              name='predict_model',
              mode=theano.compile.MonitorMode(
                  pre_func=inspect_inputs,
                  post_func=inspect_outputs)
        )

        pred_results = predict_model()

        make_submission_file(pred_results, testFnames,
                             fNameSuffix='_'.join([str(i) for i in imageShape + filterShape + poolSize + [n_hidden]]))

    else:
        pred_results = None


    return [layer0, layer1, layer2, layer3, pred_results]


def train(n_train_batches, n_valid_batches, n_test_batches,
          train_model, validate_model, test_model, print_stuff,
          n_epochs, patience = 10000, patience_increase = 2, improvement_threshold = 0.995):

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
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        print '----- epoch:', epoch
        epoch += 1

        print 'minibatch:',
        for minibatch_index in xrange(n_train_batches):

            print minibatch_index,

            if debugMode:
                print_stuff(minibatch_index)
            minibatch_avg_cost = train_model(minibatch_index)

            # print 'done evaluation train_model. minibatch average cost =', minibatch_avg_cost

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                                this_validation_loss < best_validation_loss *
                                improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

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

            if patience <= iter:
                done_looping = True
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


def read_data(xtrainfpath = '/Users/jennyyuejin/K/NDSB/Data/X_train_15_15.csv',
              xtestfpath = '/Users/jennyyuejin/K/NDSB/Data/X_test_15_15.csv',
              yfpath = '/Users/jennyyuejin/K/NDSB/Data/y.csv',
              takeNumColumns = None):

    # read training data
    x_data = np.array(pandas.read_csv(xtrainfpath, header=None), dtype=theano.config.floatX)
    y_data = np.array(pandas.read_csv(yfpath, header=None), dtype=theano.config.floatX).ravel()

    # read test data
    temp = np.array(pandas.read_csv(xtestfpath, header=None))
    testFnames = temp[:, 0]
    testData = np.array(temp[:, 1:], dtype=theano.config.floatX)

    numRows = x_data.shape[0]
    # numRows = x_data.shape[0] / batchSize * batchSize
    x_data = x_data[:numRows, :]
    y_data = y_data[:numRows]

    if takeNumColumns is not None:
        x_data = x_data[:, :takeNumColumns]
        testData = testData[:, :takeNumColumns]

    temp, testInds = StratifiedShuffleSplit(y_data, n_iter=1, test_size=0.1, random_state=1)._iter_indices().next()
    testX, testY = x_data[testInds], y_data[testInds]

    trainInds, validInds = StratifiedShuffleSplit(y_data[temp], n_iter=1, test_size=0.1, random_state=1)._iter_indices().next()
    trainX, trainY = x_data[temp][trainInds], y_data[temp][trainInds]
    validX, validY = x_data[temp][validInds], y_data[temp][validInds]

    return [(make_var(trainX), T.cast(make_var(trainY), 'int32') ),
            (make_var(validX), T.cast(make_var(validY), 'int32')),
            (make_var(testX), T.cast(make_var(testY), 'int32'))], \
           make_var(testData), \
           testFnames


if __name__ == '__main__':


    # open random image of dimensions 272 x 328
    # img = Image.open(open('/Users/jennyyuejin/K/tryTheano/Data/four.png'))
    # img_shape = (25, 25)
    #
    # # dimensions are (height, width, channel)
    # img = np.array(img.getdata(), dtype='float64') / 256.
    # img = img[: img_shape[0]*img_shape[1], 0].reshape(1, img_shape[0] * img_shape[1])

    batchSize = 1000
    trainData, testData, testFnames = read_data(xtrainfpath= '/Users/jennyyuejin/K/NDSB/Data/X_train_15_15.csv',
                                                xtestfpath = '/Users/jennyyuejin/K/NDSB/Data/X_test_15_15.csv',
                                                takeNumColumns=15*15)

    res = run_cnn(trainData,
                  121,
                  numFeatureMaps = [2, 2],
                  imageShape = [15, 15],
                  filterShape = [2, 2],
                  poolSize = [2, 2],
                  n_epochs=10, learning_rate=0.1, batch_size=batchSize, n_hidden=200,
                  predict_set_x=testData, testFnames=testFnames)


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
