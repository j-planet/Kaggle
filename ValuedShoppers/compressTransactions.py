import pandas
from dateutil import parser
import numpy as np
import itertools
import traceback
from time import time

from IterStreamer import IterStreamer
from Kaggle.pool_JJ import MyPool
from Kaggle.utilities import runPool, printDoneTime

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

# GLOBALCOUNT = 0


def read_transactions_given_id(customerId, transIndexDict, transactionsFile, headers):
    """
    @return: pandas.dataframe of transactions for the given customerId.
             If customerId doesn't exist in the index file, returns None.
    """

    # ASSUMES that everything ID in trainHistory also exists in transactions
    # if not customerId in transIndexDict.keys():
    #     return None

    temp = transIndexDict[customerId]
    startRowInd = temp[0]
    endRowInd = temp[1]

    transactionsFile.seek(0)    # reset file pointer position
    rawData = itertools.islice(transactionsFile, startRowInd, endRowInd + 1)
    res = pandas.read_csv(IterStreamer(rawData), names=headers)

    # global GLOBALCOUNT
    # GLOBALCOUNT += 1
    # print GLOBALCOUNT

    # print customerId, res.shape[0]
    return res


def chunks(l, chunkSize):
    """
    shamelessly copied from stackoverflow, again
    """
    for i in xrange(0, len(l), chunkSize):
        yield l[i : (i + chunkSize)]


def initStep(*args):
    global historyAndOffers, compEmptyDf, transDataDict
    historyAndOffers, compEmptyDf, transDataDict = args


def innerFunc(args):

    global historyAndOffers, compEmptyDf, transDataDict

    customerIds = args
    compChunk = compEmptyDf

    for customerId in customerIds:
        try:
            hao = historyAndOffers[historyAndOffers.id == customerId]    # history and offers
            # curTransData = read_transactions_given_id(customerId, transactionsIndexData,
            #                                           "/home/jj/code/Kaggle/ValuedShoppers/Data/transactions.csv")
            curTransData = transDataDict[customerId]
            # print '-----:', customerId, curTransData.shape[0]

            if curTransData is None:    # skip if there's no transactions data. unlikely
                print 'No data. skipping.'
                continue

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
            compChunk = compChunk.append(curRow, ignore_index=True)
        except:
            print('SOME ERROR HAPPENED in customer (%d): %s' % (customerId, traceback.format_exc()))

    # print 'Done with current chunk:', compChunk.shape[0], 'rows.'
    return compChunk


def create_trans_index_dict(indexFpath = "/home/jj/code/Kaggle/ValuedShoppers/Data/transIndex.csv"):

    transactionsIndexData = pandas.read_csv(indexFpath)    # id | startRowId | endRowId

    res = {}

    for i in range(transactionsIndexData.shape[0]):
        r = transactionsIndexData.irow(i)
        res[r['id']] = (r['startRowId'], r['endRowId'])

    return res


if __name__ == '__main__':
    trainHistory = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/trainHistory_wDateFields.csv")
    offers = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/offers_amended.csv")
    historyAndOffers = pandas.merge(trainHistory, offers, left_on='offer', right_on='offer', how='left')  # join train history and offers
    transIndexDict = create_trans_index_dict()

    transactionsFile = open("/home/jj/code/Kaggle/ValuedShoppers/Data/transactions.csv")
    transHeaders = transactionsFile.readline().strip().split(',')
    compTransFname = "/home/jj/code/Kaggle/ValuedShoppers/Data/transactions_train_compressed.csv"
    chunkSize_minor = 2       # number in each process
    chunkSize_major = 40    # dump every n number of rows

    freqFields = ['chain', 'category', 'company', 'brand']
    cols = [f + '_freq' for f in freqFields] \
           + [f + '_hasShopped' for f in freqFields] \
           + ['daysSinceLastPch', 'avgWeeklyPchAmt', 'pchAmtWiWeekOfOffer']

    compEmptyDf = pandas.DataFrame(columns = ['id'] + cols)
    compEmptyDf.to_csv(compTransFname, index=False)     # create new file and write columns to compressed file
    compressedTransFile = open(compTransFname, 'a')     # re-open the outputfile in append mode

    allIds = historyAndOffers.id.unique()

    for curBlockIds in chunks(allIds, chunkSize_major):

        print '======= NEW MAJOR BLOCK ========'
        # reopening the file seems to speed things up
        transactionsFile.close()
        transactionsFile = open("/home/jj/code/Kaggle/ValuedShoppers/Data/transactions.csv")

        t0 = time()
        curTransDict = {tempId: read_transactions_given_id(tempId, transIndexDict, transactionsFile, transHeaders)
                        for tempId in curBlockIds}
        print len(curBlockIds), len(curTransDict.keys())
        printDoneTime(t0, "Building transactions dict")

        to = time()
        pool = MyPool(processes=16, initializer = initStep,
                      initargs = (historyAndOffers, compEmptyDf, curTransDict))
        printDoneTime(t0, "Making the pool")

        blockOutput = runPool(pool, innerFunc, chunks(curBlockIds, chunkSize_minor))

        # dump chunk to file
        print '--- dumping ---', len(blockOutput), sum(chunk.shape[0] for chunk in blockOutput)
        for chunk in blockOutput:
            chunk.to_csv(compressedTransFile, header=False, index=False)

    compressedTransFile.close()
    transactionsFile.close()