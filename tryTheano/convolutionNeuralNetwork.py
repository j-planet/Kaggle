__author__ = 'JennyYueJin'


import cPickle
import gzip
import os
import sys
import time
import numpy
import pylab
from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
theano.config.openmp = True

sys.path.extend(['/Users/jennyyuejin/K/tryTheano'])

from logisticRegressionExample import LogisticRegression, load_data
from mlpExample import HiddenLayer


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
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
        fan_in = numpy.prod(filter_shape[1:])

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))

        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))

        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
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

        # store parameters of this layer
        self.params = [self.W, self.b]


def test_mlp(numFeatureMaps,
             imageShape,
             filterShape,
             poolSize,
             learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='/Users/JennyYueJin/K/tryTheano/Data/mnist.pkl.gz', batch_size=100,
             n_hidden=100, rndState = 0,
             predict_x = None):

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

    datasets = load_data(dataset)

    rng = numpy.random.RandomState(rndState)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, imageShape[0], imageShape[1]))

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
    layer3 = LogisticRegression(input=layer2.output, n_in=n_hidden, n_out=10)

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
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    predict_model = theano.function(
        [],
          layer3.y_pred,
          givens = {
              x: predict_x
          }
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

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train(n_train_batches, n_valid_batches, n_test_batches, train_model, validate_model, test_model, n_epochs)


    return predict_model


def train(n_train_batches, n_valid_batches, n_test_batches,
          train_model, validate_model, test_model,
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

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):

        epoch += 1

        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

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
                    test_score = numpy.mean(test_losses)

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


if __name__ == '__main__':

    model = test_mlp([2, 2], [28, 28], [5, 5], [2, 2], n_epochs=10, learning_rate=0.5,
                     predict_x=load_data('/Users/JennyYueJin/K/tryTheano/Data/mnist.pkl.gz')[0][0])

    x = T.matrix('x')   # the data is presented as rasterized images

    pred_func = theano.function(
        [],
          model.p_y_given_x,
          givens={
              x: load_data('/Users/JennyYueJin/K/tryTheano/Data/mnist.pkl.gz')[0][0]
          },
          on_unused_input='warn'
    )

    # for rndState in [0, 1, 2, 3]:
    #
    #     rng = numpy.random.RandomState(rndState)
    #
    #     # instantiate 4D tensor for input
    #     input = T.tensor4(name='input')
    #
    #     # initialize shared variable for weights.
    #     w_shp = (2, 3, 9, 9)
    #     w_bound = numpy.sqrt(3 * 9 * 9)
    #     W = theano.shared( numpy.asarray(
    #         rng.uniform(
    #             low=-1.0 / w_bound,
    #             high=1.0 / w_bound,
    #             size=w_shp),
    #         dtype=input.dtype), name ='W')
    #
    #     # initialize shared variable for bias (1D tensor) with random values
    #     # IMPORTANT: biases are usually initialized to zero. However in this
    #     # particular application, we simply apply the convolutional layer to
    #     # an image without learning the parameters. We therefore initialize
    #     # them to random values to "simulate" learning.
    #     b_shp = (2,)
    #     b = theano.shared(numpy.asarray(
    #         rng.uniform(low=-.5, high=.5, size=b_shp),
    #         dtype=input.dtype), name ='b')
    #
    #     # build symbolic expression that computes the convolution of input with filters in w
    #     conv_out = conv.conv2d(input, W)
    #
    #     # build symbolic expression to add bias and apply activation function, i.e. produce neural net layer output
    #     # A few words on ``dimshuffle`` :
    #     #   ``dimshuffle`` is a powerful tool in reshaping a tensor;
    #     #   what it allows you to do is to shuffle dimension around
    #     #   but also to insert new ones along which the tensor will be
    #     #   broadcastable;
    #     #   dimshuffle('x', 2, 'x', 0, 1)
    #     #   This will work on 3d tensors with no broadcastable
    #     #   dimensions. The first dimension will be broadcastable,
    #     #   then we will have the third dimension of the input tensor as
    #     #   the second of the resulting tensor, etc. If the tensor has
    #     #   shape (20, 30, 40), the resulting tensor will have dimensions
    #     #   (1, 40, 1, 20, 30). (AxBxC tensor is mapped to 1xCx1xAxB tensor)
    #     #   More examples:
    #     #    dimshuffle('x') -> make a 0d (scalar) into a 1d vector
    #     #    dimshuffle(0, 1) -> identity
    #     #    dimshuffle(1, 0) -> inverts the first and second dimensions
    #     #    dimshuffle('x', 0) -> make a row out of a 1d vector (N to 1xN)
    #     #    dimshuffle(0, 'x') -> make a column out of a 1d vector (N to Nx1)
    #     #    dimshuffle(2, 0, 1) -> AxBxC to CxAxB
    #     #    dimshuffle(0, 'x', 1) -> AxB to Ax1xB
    #     #    dimshuffle(1, 'x', 0) -> AxB to Bx1xA
    #     output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
    #
    #     # create theano function to compute filtered images
    #     f = theano.function([input], output)
    #
    #     # ------ fun stuff ---------
    #
    #     # open random image of dimensions 272 x 328
    #     img = Image.open(open('/Users/jennyyuejin/K/tryTheano/Data/scene.JPG'))
    #
    #     # dimensions are (height, width, channel)
    #     img = numpy.asarray(img, dtype='float64') / 256.
    #
    #     # put image in 4D tensor of shape (1, 3, height, width)
    #     img_ = img.transpose(2, 0, 1).reshape(1, 3, img.shape[0], img.shape[1])
    #     filtered_img = f(img_)
    #
    #     # plot original image and first and second components of output
    #     pylab.subplot(4, 3, rndState*3+1); pylab.axis('off'); pylab.imshow(img)
    #     pylab.gray();
    #
    #     # recall that the convOp output (filtered image) is actually a "minibatch",
    #     # of size 1 here, so we take index 0 in the first dimension:
    #     pylab.subplot(4, 3, rndState*3+2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
    #     pylab.subplot(4, 3, rndState*3+3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
    #
    # pylab.show()
    #
    #
    # input = T.dtensor4('input')
    # maxpool_shape = (2, 2)
    # pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=True)
    # f = theano.function([input],pool_out)
    #
    # invals = numpy.random.RandomState(1).rand(3, 2, 5, 5)
    # print 'With ignore_border set to True:'
    # print 'invals[0, 0, :, :] =\n', invals[0, 0, :, :]
    # print 'output[0, 0, :, :] =\n', f(invals)[0, 0, :, :]
    #
    # pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=False)
    # f = theano.function([input],pool_out)
    # print 'With ignore_border set to False:'
    # print 'invals[1, 0, :, :] =\n ', invals[1, 0, :, :]
    # print 'output[1, 0, :, :] =\n ', f(invals)[1, 0, :, :]
