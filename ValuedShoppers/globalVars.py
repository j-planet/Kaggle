from datetime import datetime
import numpy as np

HOLIDAY_TRAIN = datetime(2013, 5, 27)
HOLIDAY_TEST = datetime(2013, 7, 4)

TRANSACTIONS_FILE_NUM_LINES = 349655789
# TRANSACTIONS_FILE_NUM_LINES = 999992

DATA_DIR = "/home/jj/code/Kaggle/ValuedShoppers/Data/"

NON_X_FIELDS = ['id', 'offer', 'offerdate', 'repeattrips', 'repeater', 'repeatQuantiles']


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
            'offerPrice': False,
            'category_count': False,
            'category_avgReturnAmt': False,
            'category_totalQuantity': False,
            'category_returnRate': False,
            'category_avgPrice': False,
            'company_count': False,
            'company_avgReturnAmt': False,
            'company_totalQuantity': False,
            'company_returnRate': False,
            'company_avgPrice': False,
            'brand_count': False,
            'brand_avgReturnAmt': False,
            'brand_totalQuantity': False,
            'brand_returnRate': False,
            'brand_avgPrice': False,
            'brand_category_count': False,
            'brand_category_avgReturnAmt': False,
            'brand_category_totalQuantity': False,
            'brand_category_returnRate': False,
            'brand_category_avgPrice': False,
            'company_category_count': False,
            'company_category_avgReturnAmt': False,
            'company_category_totalQuantity': False,
            'company_category_returnRate': False,
            'company_category_avgPrice': False,
            'company_brand_count': False,
            'company_brand_avgReturnAmt': False,
            'company_brand_totalQuantity': False,
            'company_brand_returnRate': False,
            'company_brand_avgPrice': False,
            'category_company_brand_count': False,
            'category_company_brand_avgReturnAmt': False,
            'category_company_brand_totalQuantity': False,
            'category_company_brand_returnRate': False,
            'category_company_brand_avgPrice': False,
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
FIELDS_23 = ['category', 'brand', 'company', 'company_brand_avgReturnAmt', 'pchAmtWiWeekOfOffer', 'chain', 'category_totalQuantity', 'avgWeeklyPchAmt', 'company_avgReturnAmt', 'company_brand_avgPrice', 'company_category_avgPrice', 'category_avgReturnAmt', 'category_hasShopped', 'brand_avgReturnAmt', 'category_count', 'category_company_brand_avgReturnAmt', 'company_category_avgReturnAmt', 'brand_hasShopped', 'offervalue', 'brand_category_avgPrice', 'brand_category_avgReturnAmt', 'category_company_brand_avgPrice', 'company_hasShopped']

FIELDS_17 = ['category', 'pchAmtWiWeekOfOffer', 'dayOfTheWeek', 'chain_freq', 'chain', 'avgWeeklyPchAmt', 'brand', 'daysSinceLastPch', 'category_hasShopped', 'category_freq', 'brand_freq', 'brand_hasShopped', 'offervalue', 'company_freq', 'company', 'market', 'company_hasShopped']

FIELDS_10 = ['category', 'pchAmtWiWeekOfOffer', 'chain_freq', 'avgWeeklyPchAmt', 'brand', 'category_hasShopped', 'brand_hasShopped', 'offervalue', 'company', 'company_hasShopped']

COMPANY_CV_DIVISION = [[106414464],
                       [108079383, 1089520383, 1087744888],
                       [104460040, 105100050, 107717272],
                       [104610040, 107120272, 107127979, 104460040, 103320030, 104460040]]