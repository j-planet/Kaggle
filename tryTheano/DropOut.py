__author__ = 'jennyyuejin'

import numpy as np

import theano
import theano.tensor as T


class DropOut(object):

    @classmethod
    def dropOut(cls, input, dropout_rate, random_seed = 0):
        rng = np.random.RandomState(random_seed)
        # rng = np.random.RandomState(random_seed)
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))

        mask = srng.binomial(n = 1, p = 1 - dropout_rate, size = input.shape)

        # The cast is important because int * float32 = float64 which pulls things off the gpu
        return input * T.cast(mask, theano.config.floatX)


    def __init__(self, input, dropout_rate, random_seed = 0):

        self.output = DropOut.dropOut(input, dropout_rate, random_seed)