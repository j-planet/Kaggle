__author__ = 'jennyjin'

import os

ROOTDIR = '/Users/jennyjin/K/DisplayAdvertising'
DATADIR = os.path.join(ROOTDIR, 'Data')
SUBMISSIONDIR = os.path.join(ROOTDIR, 'Submissions')

ORD_COLS = ['I' + str(i) for i in range(1, 14)]
CAT_COLS = ['C' + str(i) for i in range(1, 27)]