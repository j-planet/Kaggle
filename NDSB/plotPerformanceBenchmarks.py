__author__ = 'jennyyuejin'

import re
import pandas
from matplotlib import pyplot as plt

for label, fname in \
        [('1, 3, 5', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/angle effect/small angle effect/135.txt'),
         ('+-1, +-3, +-5', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/angle effect/small angle effect/-113355.txt'),
         ('+-1, +-3', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/angle effect/small angle effect/-1133.txt'),
         ('+-1, +-2', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/angle effect/small angle effect/-1122.txt'),
         ('+-1, +-2, +-3', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/angle effect/small angle effect/-112233.txt'),
         ]:
    # [('0-seed', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/random seed effect/temp1.txt'),
    #  ('None-seed', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/random seed effect/temp0.txt'),
    #  ]:

    # ('+-1, +-3, +-5', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/temp11.txt'),for label, fname in [('90, 180', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/temp1.txt'),
    # ('-10, 10', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/temp2.txt'),
    # ('-120, 120', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/temp3.txt'),
    # ('180', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/temp4.txt'),
    # ('+-90, 180', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/temp5.txt'),
    # ('no angles', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/temp6.txt'),
    # ('-5, 5', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/temp7.txt'),
    # ('-3, 3', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/temp8.txt'),
    # ('-5, 5, 180', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/temp9.txt'),
    # ('+-3, +-5', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/temp10.txt'),
    # ('+-1, +-3, +-5', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/temp11.txt'),
    # ]:

    # for label, fname in [('(3,1,2) pool st', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/stride effect/temp1.txt'),
    #                      ('(2,1,2) pool st', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/stride effect/temp2.txt'),
    #                      ('(4,1,1) filter st (2,1,2) pool st', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/stride effect/temp3.txt'),
    #                      ('(2,1,1) filter st (2,1,2) pool st', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/stride effect/temp4.txt'),
    #                      ('(4,3,3)-(1,1,1) filter (3,1,2) pool st', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/stride effect/temp5.txt'),
    #                      ('(3,3,3)-(1,1,1) filter (3,1,3) pool st', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/stride effect/temp6.txt'),
    #                      ('(2,3,3)-(1,1,1) filter (3,1,3) pool st', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/stride effect/temp7.txt'),
    #                      ('original', '/Users/jennyyuejin/K/NDSB/Data/Performance benchmarks/temp6.txt')
    #                      ]:

    print fname
    epochs = []
    errs = []

    for line in open(fname):

        if not line.startswith('epoch'):
            continue

        temp = re.findall(r'[.\w]+', line)
        epochNum = float(temp[1])
        err = float(temp[7])

        epochs.append(epochNum)
        errs.append(err)

    df = pandas.DataFrame({'epoch': epochs, 'err': errs}).pivot_table(columns='epoch', values='err').reset_index()

    plt.plot(df['epoch'], df['err'], '.-', label=label)

plt.legend()
plt.show()
