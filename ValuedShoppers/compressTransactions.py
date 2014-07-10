import pandas
from dateutil import parser
import numpy as np
import itertools
import traceback
from time import time, sleep

from IterStreamer import IterStreamer
from Kaggle.pool_JJ import MyPool
from Kaggle.utilities import runPool, printDoneTime


def read_transactions_given_id(customerId, transIndexDict, transactionsFile, headers):
    """
    @return: pandas.dataframe of transactions for the given customerId.
             If customerId doesn't exist in the index file, returns None.
    """

    # ASSUMES that everything ID in trainHistory also exists in transactions

    # startRowInd = 2
    # endRowInd = 3000
    # t0 = time()
    startRowInd, endRowInd = transIndexDict[customerId]
    # printDoneTime(t0, 'startEndRowId')


    transactionsFile.seek(0)    # reset file pointer position
    rawData = itertools.islice(transactionsFile, startRowInd, endRowInd + 1)
    # t0 = time()
    res = pandas.read_csv(IterStreamer(rawData), names=headers)
    # print '----', int(100000. * (time()-t0)/res.shape[0])

    return res
    # sleep(0.5)
    # return testRes
    # return None

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
            curTransData = transDataDict[customerId]

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

    return compChunk


if __name__ == '__main__':
    trainHistory = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/trainHistory_wDateFields.csv")
    offers = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/offers_amended_1.csv")
    historyAndOffers = pandas.merge(trainHistory, offers, left_on='offer', right_on='offer', how='left')  # join train history and offers
    transIndexData = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/transIndex.csv")
    compTransFname = "/home/jj/code/Kaggle/ValuedShoppers/Data/transactions_train_compressed.csv"
    chunkSize_minor = 250    # number in each process
    chunkSize_major = 4000   # dump every n number of rows

    freqFields = ['chain', 'category', 'company', 'brand']
    cols = [f + '_freq' for f in freqFields] \
           + [f + '_hasShopped' for f in freqFields] \
           + ['daysSinceLastPch', 'avgWeeklyPchAmt', 'pchAmtWiWeekOfOffer']

    compEmptyDf = pandas.DataFrame(columns = ['id'] + cols)
    compEmptyDf.to_csv(compTransFname, index=False)     # create new file and write columns to compressed file
    compressedTransFile = open(compTransFname, 'a')     # re-open the outputfile in append mode

    transactionsFile = open("/home/jj/code/Kaggle/ValuedShoppers/Data/transactions.csv")
    transHeaders = transactionsFile.readline().strip().split(',')
    blockTransDict = {}
    print '======= NEW MAJOR BLOCK ========'

    t_dict = time()

    for rowNum in range(transIndexData.shape[0]):

        # ----- keep building transactions data for this major block
        customerId, startRowId, endRowId = transIndexData.irow(rowNum)
        numRows = endRowId - startRowId + 1
        rawData = [transactionsFile.next() for _ in range(numRows)]     # assumes the row numbers in the index file are adjacent

        if customerId in np.array(historyAndOffers.id):
            # print customerId, 'is in train!!'
            blockTransDict[customerId] = pandas.read_csv(IterStreamer(rawData), names = transHeaders)
            # print len(blockTransDict)

        # ---- finished building. run the pool for this major block
        if len(blockTransDict) == chunkSize_major or rowNum == transIndexData.shape[0]-1:

            totalTime = time() - t_dict
            print "Building transactions dict total:", totalTime
            print "Building transactions dict each:", 1000000.* totalTime/ sum(df.shape[0] for df in blockTransDict.values())

            print '--------- Finished building. Running pool. --------'

            t0 = time()
            pool = MyPool(processes=16, initializer = initStep,
                          initargs = (historyAndOffers, compEmptyDf, blockTransDict))
            printDoneTime(t0, "Making the pool")

            t0 = time()
            poolOutputs = runPool(pool, innerFunc, chunks(blockTransDict.keys(), chunkSize_minor))
            printDoneTime(t0, 'Running the pool')

            # dump pool output to file
            for chunk in poolOutputs:
                chunk.to_csv(compressedTransFile, header=False, index=False)

            print '--- dumping ---', len(poolOutputs), sum(chunk.shape[0] for chunk in poolOutputs)

            pool.close()
            pool.join()
            pool.terminate()

            # reset major block dict
            print '======= NEW MAJOR BLOCK ========'
            blockTransDict = {}
            t_dict = time()


    compressedTransFile.close()
    transactionsFile.close()