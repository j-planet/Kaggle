from datetime import datetime
import numpy as np

HOLIDAY_TRAIN = datetime(2013, 5, 27)
HOLIDAY_TEST = datetime(2013, 7, 4)

TRANSACTIONS_FILE_NUM_LINES = 349655790

DATA_DIR = "/home/jj/code/Kaggle/ValuedShoppers/Data/"

# key: field name; value: if discrete
X_FIELDS = {'chain': True,
           'market': True,
           'dayOfTheWeek': True,
           'isWeekend': True,
           'isHoliday': True,
           'isAWeekBeforeHol': True,
           'isTwoWeeksBeforeHol': True,
           'isAWeekAfterHol': True,
           'isTwoWeeksAfterHol': True,
           'category': True,
           'quantity': False,
           'company': True,
           'offervalue': False,
           'brand': True,
           'chain_freq': False,
           'category_freq': False,
           'company_freq': False,
           'brand_freq': False,
           'chain_hasShopped': True,
           'category_hasShopped': True,
           'company_hasShopped': True,
           'brand_hasShopped': True,
           'daysSinceLastPch': False,
           'avgWeeklyPchAmt': False,
           'pchAmtWiWeekOfOffer': False}

FIELDS_17 = ['category', 'pchAmtWiWeekOfOffer', 'dayOfTheWeek', 'chain_freq', 'chain', 'avgWeeklyPchAmt', 'brand', 'daysSinceLastPch', 'category_hasShopped', 'category_freq', 'brand_freq', 'brand_hasShopped', 'offervalue', 'company_freq', 'company', 'market', 'company_hasShopped']

FIELDS_10 = ['category', 'pchAmtWiWeekOfOffer', 'chain_freq', 'avgWeeklyPchAmt', 'brand', 'category_hasShopped', 'brand_hasShopped', 'offervalue', 'company', 'company_hasShopped']