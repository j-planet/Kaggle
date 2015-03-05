__author__ = 'jennyyuejin'

import re
from matplotlib import pyplot as plt

for label, fname in [('-10, 10', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/temp1.txt'),
                     ('90, 180', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/temp2.txt'),
                     ('-120, 120', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/temp3.txt')]:

    print fname
    epochs = []
    errs = []

    for line in open(fname):

        if not line.startswith('epoch'):
            continue

        temp = re.findall(r'[.\w]+', line)
        epochNum = temp[1]
        err = temp[7]

        epochs.append(epochNum)
        errs.append(err)

    plt.plot(epochs, errs, '.-', label=label)

plt.legend()
plt.show()
