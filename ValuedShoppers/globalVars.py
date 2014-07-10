from datetime import datetime
import numpy as np

HOLIDAY_TRAIN = datetime(2013, 5, 27)
HOLIDAY_TEST = datetime(2013, 7, 4)

TRANSACTIONS_FILE_NUM_LINES = 349655789
# TRANSACTIONS_FILE_NUM_LINES = 999992

DATA_DIR = "/home/jj/code/Kaggle/ValuedShoppers/Data/"

NON_X_FIELDS = ['id', 'offer', 'offerdate', 'repeattrips', 'repeater', 'repeatQuantiles']


# key: field name; value: if discrete
X_FIELDS = ['company_category_totalQuantity',
            'chain',
            'brand_avgPrice',
            'brand_avgRepeatPerUser',
            'company_brand_totalQuantity',
            'category_avgRepeatPerRepeatUser',
            'offerPrice',
            'company_brand_repeatFreq',
            'dayOfTheWeek',
            'category_company_brand_repeatFreq',
            'isTwoWeeksAfterHol',
            'company_category_avgRepeatPerRepeatUser',
            'avgRepeatPerUser',
            'brand_hasShopped',
            'brand_category_avgRepeatPerUser',
            'brand_pctDiscount',
            'brand_avgReturnAmt',
            'avgRepeatPerRepeatUser',
            'brand_category_avgPrice',
            'category_avgPrice',
            'company_avgRepeatPerRepeatUser',
            'brand_returnRate',
            'avgWeeklyPchAmt',
            'company_avgReturnAmt',
            'category_avgReturnAmt',
            'isWeekend',
            'category_company_brand_totalQuantity',
            'company_brand_avgReturnAmt',
            'category_company_brand_pctDiscount',
            'brand_category_avgRepeatPerRepeatUser',
            'offervalue',
            'company_brand_avgRepeatPerRepeatUser',
            'brand_category_totalQuantity',
            'company_brand_returnRate',
            'company_brand_count',
            'isHoliday',
            'company_category_repeatFreq',
            'category_company_brand_avgPrice',
            'isAWeekBeforeHol',
            'brand_freq',
            'brand_category_returnRate',
            'chain_hasShopped',
            'brand_category_repeatFreq',
            'category_repeatFreq',
            'company_brand_avgRepeatPerUser',
            'category_avgRepeatPerUser',
            'isAWeekAfterHol',
            'brand_category_avgReturnAmt',
            'market',
            'company_avgPrice',
            'repeatFreq',
            'chain_freq',
            'category_totalQuantity',
            'category_company_brand_avgReturnAmt',
            'category_pctDiscount',
            'category_freq',
            'company_category_avgRepeatPerUser',
            'brand_count',
            'company_category_pctDiscount',
            'brand',
            'company_totalQuantity',
            'category_company_brand_count',
            'category_company_brand_avgRepeatPerRepeatUser',
            'company_category_avgPrice',
            'company_category_avgReturnAmt',
            'company_repeatFreq',
            'company_category_returnRate',
            'company_freq',
            'pchAmtWiWeekOfOffer',
            'company_hasShopped',
            'company_brand_pctDiscount',
            'company_count',
            'brand_category_count',
            'company_brand_avgPrice',
            'brand_category_pctDiscount',
            'isTwoWeeksBeforeHol',
            'company_category_count',
            'company',
            'category',
            'brand_avgRepeatPerRepeatUser',
            'company_returnRate',
            'category_company_brand_returnRate',
            'category_count',
            'category_company_brand_avgRepeatPerUser',
            'brand_totalQuantity',
            'company_avgRepeatPerUser',
            'brand_repeatFreq',
            'daysSinceLastPch',
            'category_hasShopped',
            'category_returnRate',
            'company_pctDiscount',
            'quantity']
FIELDS_23 = ['category', 'brand', 'company', 'company_brand_avgReturnAmt', 'pchAmtWiWeekOfOffer', 'chain', 'category_totalQuantity', 'avgWeeklyPchAmt', 'company_avgReturnAmt', 'company_brand_avgPrice', 'company_category_avgPrice', 'category_avgReturnAmt', 'category_hasShopped', 'brand_avgReturnAmt', 'category_count', 'category_company_brand_avgReturnAmt', 'company_category_avgReturnAmt', 'brand_hasShopped', 'offervalue', 'brand_category_avgPrice', 'brand_category_avgReturnAmt', 'category_company_brand_avgPrice', 'company_hasShopped']

FIELDS_17 = ['category', 'pchAmtWiWeekOfOffer', 'dayOfTheWeek', 'chain_freq', 'chain', 'avgWeeklyPchAmt', 'brand', 'daysSinceLastPch', 'category_hasShopped', 'category_freq', 'brand_freq', 'brand_hasShopped', 'offervalue', 'company_freq', 'company', 'market', 'company_hasShopped']

FIELDS_10 = ['category', 'pchAmtWiWeekOfOffer', 'chain_freq', 'avgWeeklyPchAmt', 'brand', 'category_hasShopped', 'brand_hasShopped', 'offervalue', 'company', 'company_hasShopped']

COMPANY_CV_DIVISION = [[106414464],
                       [108079383, 1089520383, 1087744888],
                       [104460040, 105100050, 107717272],
                       [104610040, 107120272, 107127979, 104460040, 103320030, 104460040]]