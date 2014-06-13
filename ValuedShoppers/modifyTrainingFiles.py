import pandas
from dateutil import parser
import numpy as np

from globalVars import *


def add_dates_to_history():
    for isTraining in [True, False]:

        data = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/" +
                               ("train" if isTraining else "test") + "History.csv")
        # isTraining = True
        holiday = (HOLIDAY_TRAIN if isTraining else HOLIDAY_TEST)

        offerdates = [parser.parse(d) for d in data.offerdate]  # convert to datetime objects
        dayOfTheWeek = [d.weekday() for d in offerdates]
        isWeekend = (np.array(dayOfTheWeek) >= 5)                         # Saturday=5; Sunday=6
        isHoliday = [d == holiday for d in offerdates]
        daysAfterHoliday = np.array([(d - holiday).days for d in offerdates])
        isAWeekBeforeHol = np.logical_and(daysAfterHoliday<0, daysAfterHoliday>-7)
        isTwoWeeksBeforeHol = np.logical_and(daysAfterHoliday<0, daysAfterHoliday>-14)
        isAWeekAfterHol = np.logical_and(daysAfterHoliday>0, daysAfterHoliday<7)
        isTwoWeeksAfterHol = np.logical_and(daysAfterHoliday>0, daysAfterHoliday<14)

        data['dayOfTheWeek'] = dayOfTheWeek
        data['isWeekend'] = isWeekend
        data['isHoliday'] = isHoliday
        data['isAWeekBeforeHol'] = isAWeekBeforeHol
        data['isTwoWeeksBeforeHol'] = isTwoWeeksBeforeHol
        data['isAWeekAfterHol'] = isAWeekAfterHol
        data['isTwoWeeksAfterHol'] = isTwoWeeksAfterHol

        # del data['offerdate']

        data.to_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/" +
                    ("train" if isTraining else "test") + "History_wDateFields.csv", index=False)


