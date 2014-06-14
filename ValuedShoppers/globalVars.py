from datetime import datetime

HOLIDAY_TRAIN = datetime(2013, 5, 27)
HOLIDAY_TEST = datetime(2013, 7, 4)

DATA_DIR = "/home/jj/code/Kaggle/ValuedShoppers/Data/"

X_FIELDS = ['chain',
           'market',
           'dayOfTheWeek',
           'isWeekend',
           'isHoliday',
           'isAWeekBeforeHol',
           'isTwoWeeksBeforeHol',
           'isAWeekAfterHol',
           'isTwoWeeksAfterHol',
           'category',
           'quantity',
           'company',
           'offervalue',
           'brand',
           'chain_freq',
           'category_freq',
           'company_freq',
           'brand_freq',
           'chain_hasShopped',
           'category_hasShopped',
           'company_hasShopped',
           'brand_hasShopped',
           'daysSinceLastPch',
           'avgWeeklyPchAmt',
           'pchAmtWiWeekOfOffer']