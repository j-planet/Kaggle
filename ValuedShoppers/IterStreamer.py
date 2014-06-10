# shamelessly copied from http://stackoverflow.com/questions/6657820/python-convert-an-iterable-to-a-stream


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
        data = self.leftover
        count = len(self.leftover)
        try:
            while count < size:
                chunk = self.next()
                data += chunk
                count += len(chunk)
        except StopIteration, e:
            self.leftover = ''
            return data

        if count > size:
            self.leftover = data[size:]

        return data[:size]
