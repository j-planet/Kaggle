import pandas
from dateutil import parser
import numpy as np
import itertools

from IterStreamer import IterStreamer

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

def read_transactions_given_id(customerId,
                               transIndexData,
                               transFpath = "/home/jj/code/Kaggle/ValuedShoppers/Data/transactions_small.csv"):
    """
    @return: pandas.dataframe of transactions for the given customerId.
             If customerId doesn't exist in the index file, returns None.
    """

    transactionsFile = open(transFpath)
    headers = transactionsFile.readline().strip().split(',')

    # If customerId doesn't exist in the index file, returns None.
    if not customerId in list(transIndexData.id):
        return None

    posInIndexFile = list(transIndexData.id).index(customerId)
    startRowInd = transIndexData.startRowId.irow(posInIndexFile)
    endRowInd = transIndexData.endRowId.irow(posInIndexFile)
    rawData = itertools.islice(transactionsFile, startRowInd, endRowInd + 1)
    res = pandas.read_csv(IterStreamer(rawData), names=headers)

    transactionsFile.close()
    return res

trainHistory = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/trainHistory_wDateFields.csv")
offers = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/offers_amended.csv")
historyAndOffers_train = pandas.merge(trainHistory, offers, left_on='offer', right_on='offer', how='left')  # join train history and offers
transactionsIndexData = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/transIndex_small.csv")    # id | startRowId | endRowId
compTransFname = "/home/jj/code/Kaggle/ValuedShoppers/Data/transactions_small_compressed.csv"
outputChunkSize = 20   # dump every n number of rows

freqFields = ['chain', 'category', 'company', 'brand']
cols = [f + '_freq' for f in freqFields] \
       + [f + '_hasShopped' for f in freqFields] \
       + ['daysSinceLastPch', 'avgWeeklyPchAmt', 'pchAmtWiWeekOfOffer']

compEmptyDf = pandas.DataFrame(columns = ['id'] + cols)
compEmptyDf.to_csv(compTransFname, index=False)     # create new file and write columns to compressed file
compressedTransFile = open(compTransFname, 'a')     # re-open the outputfile in append mode


allIds = historyAndOffers_train.id.unique()[:975]
compChunk = compEmptyDf

for i, customerId in enumerate(allIds):
    # id = 86246

    hao = historyAndOffers_train[historyAndOffers_train.id == customerId]    # history and offers
    curTransData = read_transactions_given_id(customerId, transactionsIndexData,
                                              "/home/jj/code/Kaggle/ValuedShoppers/Data/transactions_small.csv")

    if curTransData is None:    # skip if there's no transactions data. unlikely
        continue

    print '-----', i, 'out of', len(allIds),':', customerId

    # ---- frequency (has shopped, frequency of shopping at a chain, for example)
    curRow = {'id': customerId}
    for freqField in freqFields:

        numShopped = (curTransData[freqField] == np.array(hao[freqField])[0]).sum()
        hasShopped = numShopped > 0
        freq = 1. * numShopped / curTransData.shape[0]

        curRow[freqField + '_freq'] = freq
        curRow[freqField + '_hasShopped'] = hasShopped

    # ---- time since last shopping date
    dates = [parser.parse(d) for d in curTransData.date]  # convert to datetime objects
    curOfferDate = parser.parse(np.array(hao.offerdate)[0])
    curRow['daysSinceLastPch'] = min([abs(d - curOfferDate) for d in dates]).days

    # ---- average weekly purchase amount
    curRow['avgWeeklyPchAmt'] = curTransData.purchaseamount.sum() / (max(dates)-min(dates)).days * 7

    #---- amount purchased within a week of offer
    curRow['pchAmtWiWeekOfOffer'] = curTransData.purchaseamount[[abs(d - curOfferDate).days < 7 for d in dates]].sum()

    # add to the chunk
    # pandas.DataFrame(curRow).to_csv(compressedTransFile, header=False)
    compChunk = compChunk.append(curRow, ignore_index=True)

    # dump chunk to file
    if compChunk.shape[0] == outputChunkSize:
        print 'dumping'
        compChunk.to_csv(compressedTransFile, header=False, index=False)
        compChunk = compEmptyDf     # clear chunk

compChunk.to_csv(compressedTransFile, header=False, index=False)    # dump leftover chunk
compressedTransFile.close()