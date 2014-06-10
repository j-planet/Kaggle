import pandas
from dateutil import parser
import numpy as np

# --------- add dates to transactions ---------------
# dates = [parser.parse(d) for d in transactions.date]  # convert to datetime objects

# dayOfTheWeek = [d.weekday() for d in dates]
# isWeekend = (np.array(dayOfTheWeek) >= 5)                         # Saturday=5; Sunday=6
# days = [d.day for d in dates]
# months = [d.month for d in dates]
#
# transactions['dayOfTheWeek'] = dayOfTheWeek
# transactions['isWeekend'] = isWeekend
# transactions['days'] = days
# transactions['months'] = months

trainHistory = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/trainHistory_wDateFields.csv")
transactions = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/transactions_small.csv")
offers = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/offers_amended.csv")
historyAndOffers_train = pandas.merge(trainHistory, offers, left_on='offer', right_on='offer', how='left')  # join train history and offers

freqFields = ['chain', 'category', 'company', 'brand']
cols = [f + '_freq' for f in freqFields] + [f + '_hasShopped' for f in freqFields] \
       + ['daysSinceLastPch', 'avgWeeklyPchAmt', 'pchAmtWiWeekOfOffer']
res = pandas.DataFrame(columns = ['id'] + cols)
allIds = transactions.id.unique()


for i, id in enumerate(allIds):

    hao = historyAndOffers_train[historyAndOffers_train.id == id]    # history and offers

    # continue if there's no train history
    if hao.shape[0] == 0:
        continue

    print '-----', i, 'out of', len(allIds),':', id
    # id = 86246

    t = transactions[transactions.id == id]

    # ---- frequency (has shopped, frequency of shopping at a chain, for example)
    curRow = {'id': id}
    for freqField in freqFields:

        numShopped = (t[freqField] == np.array(hao[freqField])[0]).sum()
        hasShopped = numShopped > 0
        freq = 1. * numShopped / t.shape[0]

        curRow[freqField + '_freq'] = freq
        curRow[freqField + '_hasShopped'] = hasShopped

    # ---- time since last shopping date
    dates = [parser.parse(d) for d in t.date]  # convert to datetime objects
    curOfferDate = parser.parse(np.array(hao.offerdate)[0])
    curRow['daysSinceLastPch'] = min([abs(d - curOfferDate) for d in dates]).days

    # ---- average weekly purchase amount
    curRow['avgWeeklyPchAmt'] = t.purchaseamount.sum() / (max(dates)-min(dates)).days * 7

    #---- amount purchased within a week of offer
    curRow['pchAmtWiWeekOfOffer'] = t.purchaseamount[[abs(d - curOfferDate).days < 7 for d in dates]].sum()

    # add to result
    res = res.append(curRow, ignore_index=True)

res.to_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/transactions_compressed.csv")