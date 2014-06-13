import pandas
from dateutil import parser
import numpy as np
import os

from globalVars import *


def convert_to_ranks(s):
    """
    converts a series of values into their corresponding ranks (the higher the larger rank)
    @param s: pandas series
    """
    return [np.sum(s < d) for d in s]


def modify_training_history(isTraining):

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

    data.to_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/" +
                ("train" if isTraining else "test") + "History_wDatesQuantiles.csv", index=False)


def add_quantiles_to_training_history(inputFpath):

    data = pandas.read_csv(inputFpath)

    print 'read. adding quantiles.'
    data['repeatQuantiles'] = convert_to_ranks(data['repeattrips'])

    data.to_csv(os.path.join(DATA_DIR, "trainHistory_wDatesQuantiles.csv"))


if __name__=='__main__':
    add_quantiles_to_training_history("/home/jj/code/Kaggle/ValuedShoppers/Data/trainHistory_wDateFields.csv")