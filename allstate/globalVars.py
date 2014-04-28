import numpy as np

DATA_DIR = '/home/jj/code/Kaggle/allstate/Data/'
CONDENSED_TABLES_DIR = '/home/jj/code/Kaggle/allstate/condensedTables/'

CAR_VALUE_MAPPING = dict(zip(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], np.arange(1, 10)))
IND_COL = u'customer_ID'
OUTPUT_COLS = [u'A', u'B', u'C', u'D', u'E', u'F', u'G']
