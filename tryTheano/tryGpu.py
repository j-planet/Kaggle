__author__ = 'JennyYueJin'

from theano import function, config, shared, sandbox
import theano
import theano.tensor as T
import numpy
import time
import os

print theano.config.device
print os.environ['THEANO_FLAGS']



class Klass(object):

    def __init__(self):
        self.x = T.matrix('x')
        self.res = T.exp(self.x)

    def calc(self, xVal):

        # x = T.dmatrix('x')

        f = theano.function(
            [],
              self.res,
              givens = {
                  self.x: xVal
              },
              on_unused_input='warn'
        )

        return f()


if __name__ == '__main__':

    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function(
        [],
          T.exp(x),
          mode='FAST_RUN')
    print f.maker.fgraph.toposort()

    t0 = time.time()
    for i in xrange(iters):
        r = f()
    t1 = time.time()

    print 'Looping %d times took' % iters, t1 - t0, 'seconds'
    print 'Result is', r
    if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print 'Used the cpu'
    else:
        print 'Used the gpu'

    # print Klass().calc([[0, 1], [-1, -2]])