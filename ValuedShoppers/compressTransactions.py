import pandas

import numpy as np
import itertools
from itertools import chain
import traceback
from time import time, sleep
from dateutil import parser
from datetime import timedelta
from os import path

from IterStreamer import IterStreamer
from Kaggle.pool_JJ import MyPool
from Kaggle.utilities import runPool, printDoneTime

from globalVars import *


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


def _processBlockMetrics(transData, ind):
    """
    processes a chunk of transaction data
    @param ind: array of booleans, used for filtering the given transaction data
    @param transData: block of transaction data (pandas DF)
    @return: a dictionary
    """

    numShopped = ind.sum()
    hasShopped = numShopped > 0
    freq = 1. * numShopped / transData.shape[0]
    amt = sum(transData['purchaseamount'][ind])      # total purchase amount
    quant = sum(transData['purchasequantity'][ind])      # total purchase quantity

    return {'numShopped': numShopped,
            'neverShopped': not hasShopped,
            'freq': freq,
            'hasShopped': hasShopped,
            'amt': amt,
            'quant': quant
    }


def _processField(transData, hao, fieldName, ind):
    """
    processes a chunk of transaction data
    @param hao: history and offer data (assumed to be one row)
    @param transData: block of transaction data (pandas DF)
    @param ind: array of booleans, used for filtering the given transaction data specific to the field (e.g. category)
    @return: a dictionary
    """

    res = {}

    tempD = _processBlockMetrics(transData, ind)     # process block
    res.update({'_'.join([fieldName, k]): v for k, v in tempD.iteritems()})    # update with block results

    # ----- day-specific fields
    curTransDates = np.array([parser.parse(d) for d in transData['date']])
    curOfferDate = parser.parse(hao['offerdate'].irow(0))
    datesBeforeInd = (curTransDates <= curOfferDate)

    for daysGap in [7, 30, 60, 180]:
        ind_dates = np.logical_and(ind, datesBeforeInd,
                                   curTransDates >= curOfferDate - timedelta(days=daysGap))
        tempD = _processBlockMetrics(transData, ind_dates)     # process block
        res.update({'_'.join([fieldName, k, str(daysGap)]): v for k, v in tempD.iteritems()})    # update with block results

    return res


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

            # --- single fields
            for freqField in freqFields_single:
                ind = (curTransData[freqField] == np.array(hao[freqField])[0])
                curRow.update(_processField(curTransData, hao, freqField, ind))

            # --- double fields
            for f1, f2 in freqFields_double:
                ind = np.logical_and(curTransData[f1] == np.array(hao[f1])[0],
                                     curTransData[f2] == np.array(hao[f2])[0])
                curRow.update(_processField(curTransData, hao, f1 + '_' + f2, ind))


            # --- triple field
            for f1, f2, f3 in freqFields_triple:    # altho there's only one triplet for now
                ind = np.logical_and(curTransData[f1] == np.array(hao[f1])[0],
                                     curTransData[f2] == np.array(hao[f2])[0],
                                     curTransData[f3] == np.array(hao[f3])[0])
                curRow.update(_processField(curTransData, hao, f1 + '_' + f2 + '_' + f3, ind))

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


def flatten(lol):
    """
    flattens a list of lists
    """

    return list(chain.from_iterable(lol))


def makeOutputColumnHeaders(features, daysVec):
    """
    make columns for the resulting compressed transactions file
    @param features: e.g. ['numShopped', 'freq', 'hasShopped', 'neverShopped', 'amt', 'quant']
    @param daysVec: [7, 30, 60, 180]
    """

    temp = flatten([ [ '_'.join([f, b]) for b in features] for f in freqFields_single]) \
           + flatten([ ['_'.join([f1, f2, b]) for b in features] for f1, f2 in freqFields_double]) \
           + flatten([ ['_'.join([f1, f2, f3, b]) for b in features] for f1, f2, f3 in freqFields_triple])

    return temp + flatten([[k + '_' + str(d) for k in temp] for d in daysVec]) + ['daysSinceLastPch', 'avgWeeklyPchAmt', 'pchAmtWiWeekOfOffer']


if __name__ == '__main__':
    trainHistory = pandas.read_csv(path.join(DATA_DIR, "trainHistory_wDateFields.csv"))
    offers = pandas.read_csv(path.join(DATA_DIR, "offers.csv"))
    historyAndOffers = pandas.merge(trainHistory, offers, left_on='offer', right_on='offer', how='left')  # join train history and offers
    # transIndexData = pandas.read_csv(path.join(DATA_DIR, "transIndex_small.csv"))
    transIndexData = pandas.read_csv(path.join(DATA_DIR, "transIndex.csv"))
    compTransFname = path.join(DATA_DIR, "transactions_train_compressed_v2.csv")
    # compTransFname = path.join(DATA_DIR, "temp.csv")
    chunkSize_minor = 250    # number in each process
    chunkSize_major = 4000   # dump every n number of rows

    freqFields_single = ['chain', 'category', 'company', 'brand']
    freqFields_double = [('category', 'company'), ('category', 'brand'), ('company', 'brand')]
    freqFields_triple = [('category', 'company', 'brand')]

    cols = makeOutputColumnHeaders(features= ['numShopped', 'freq', 'hasShopped', 'neverShopped', 'amt', 'quant'],
                                   daysVec=[7, 30, 60, 180])


    compEmptyDf = pandas.DataFrame(columns = ['id'] + cols)
    compEmptyDf.to_csv(compTransFname, index=False)     # create new file and write columns to compressed file
    compressedTransFile = open(compTransFname, 'a')     # re-open the outputfile in append mode

    # transactionsFile = open(path.join(DATA_DIR, "transactions_small.csv"))
    transactionsFile = open(path.join(DATA_DIR, "transactions.csv"))
    transHeaders = transactionsFile.readline().strip().split(',')
    blockTransDict = {}
    print '======= NEW MAJOR BLOCK ========'

    t_dict = time()

    for rowNum in range(transIndexData.shape[0]):
        # print '>>>>> %d out of %d rows in transIndexData:', rowNum, transIndexData.shape[0]

        # ----- keep building transactions data for this major block
        customerId, startRowId, endRowId = transIndexData.irow(rowNum)
        numRows = endRowId - startRowId + 1
        rawData = [transactionsFile.next() for _ in range(numRows)]     # assumes the row numbers in the index file are adjacent

        if customerId in np.array(historyAndOffers.id):
            blockTransDict[customerId] = pandas.read_csv(IterStreamer(rawData), names = transHeaders)

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