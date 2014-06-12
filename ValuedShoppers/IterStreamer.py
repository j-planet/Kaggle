# shamelessly copied and modified from http://stackoverflow.com/questions/6657820/python-convert-an-iterable-to-a-stream
from time import time
from Kaggle.utilities import printDoneTime

class IterStreamer(object):
    """
    File-like streaming iterator.
    """
    def __init__(self, generator):
        self.generator = generator
        self.iterator = iter(generator)
        self.leftover = ''

    def __len__(self):
        return self.generator.__len__()

    def __iter__(self):
        return self.iterator

    def next(self):
        return self.iterator.next()

    def read(self, size):

        count = len(self.leftover)
        data = self.leftover

        # ----- does not have enough in leftover
        if count < size:
            try:
                while count < size:
                    chunk = self.next()
                    data += chunk
                    count += len(chunk)
            except StopIteration, e:
                pass

        self.leftover = data[size:]

        return data[:size]

