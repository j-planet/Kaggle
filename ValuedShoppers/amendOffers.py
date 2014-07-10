import pandas
import numpy as np
from pprint import pprint

from globalVars import *
from IterStreamer import IterStreamer

offers = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/offers.csv")
transactionsFile = open("/home/jj/code/Kaggle/ValuedShoppers/Data/transactions.csv")
transHeaders = transactionsFile.readline().strip().split(',')
trainHistoryData = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/trainHistory_wDatesQuantiles.csv")


def _process_chunk(chunk, fieldInd, negQuantInd, posQuantInd, res):
    res['count'] += fieldInd.sum()
    # price
    ind = fieldInd & posQuantInd
    res['avgPrice'] += 1. * (
    chunk.purchaseamount[ind] / chunk.purchasequantity[ind]).sum()    # really the total for now
    # popularities
    res['totalQuantity'] += chunk.purchasequantity[fieldInd].sum()
    # number of returns
    res['returnRate'] += (negQuantInd & fieldInd).sum()     # really the count for now
    res['avgReturnAmt'] += chunk.purchaseamount[negQuantInd & fieldInd].sum()   # really the total for now


def preprocess_single_dict(_singleFieldDict, chunk, posQuantInd, negQuantInd):

    for fieldName, d in _singleFieldDict.iteritems():

        for fieldValue, res in d.iteritems():

            fieldInd = (chunk[fieldName] == fieldValue)
            _process_chunk(chunk, fieldInd, negQuantInd, posQuantInd, res)


def preprocess_double_dict(_doubleFieldDict, chunk, posQuantInd, negQuantInd):

    for fieldNames, d in _doubleFieldDict.iteritems():

        for fieldValues, res in d.iteritems():

            fieldInd = np.logical_and(chunk[fieldNames[0]] == fieldValues[0], chunk[fieldNames[1]] == fieldValues[1])
            _process_chunk(chunk, fieldInd, negQuantInd, posQuantInd, res)


def preprocess_triple_dict(_tripleFieldDict, chunk, posQuantInd, negQuantInd):

    fieldNames = ['category', 'company', 'brand']
    for fieldValues, res in _tripleFieldDict.iteritems():

        fieldInd = np.logical_and(chunk[fieldNames[0]] == fieldValues[0], chunk[fieldNames[1]] == fieldValues[1], chunk[fieldNames[2]] == fieldValues[2])
        _process_chunk(chunk, fieldInd, negQuantInd, posQuantInd, res)


def _postprocess(d):
    for fieldValue, res in d.iteritems():
        res['avgPrice'] = res['avgPrice'] / res['count'] if res['count'] > 0 else np.nan
        res['avgReturnAmt'] = res['avgReturnAmt'] / res['returnRate'] if res[
                                                                             'returnRate'] > 0 else np.nan    # returnRate is just a count for now. it's important to do this before the next line
        res['returnRate'] = res['returnRate'] / res['count'] if res['count'] > 0 else np.nan


def inner_loop(numLinesToRead, sd, dd, td):

    rawData = [transactionsFile.next() for _ in xrange(numLinesToRead)]
    chunk = pandas.read_csv(IterStreamer(rawData), names=transHeaders)

    posQuantInd = (chunk.purchasequantity > 0)
    negQuantInd = np.logical_not(posQuantInd)

    preprocess_single_dict(chunk, posQuantInd, negQuantInd)
    preprocess_double_dict(chunk, posQuantInd, negQuantInd)
    preprocess_triple_dict(chunk, posQuantInd, negQuantInd)


def write_singleDict_to_offers(_offers, _singleFieldDict):
    partialNames = _singleFieldDict.values()[0].values()[0].keys()

    for fieldName, d in _singleFieldDict.iteritems():

        for partialName in partialNames:
            _offers[fieldName + '_' + partialName] = None

        for fieldValue, res in d.iteritems():
            ind = _offers[fieldName] == fieldValue

            for k, v in res.iteritems():
                _offers[fieldName + '_' + k][ind] = v


def write_doubleDict_to_offers(_offers, _doubleFieldDict):
    partialNames = _doubleFieldDict.values()[0].values()[0].keys()

    for fieldNames, d in _doubleFieldDict.iteritems():

        for partialName in partialNames:
            _offers['_'.join(list(fieldNames) + [partialName])] = None

        for fieldValues, res in d.iteritems():

            ind = np.logical_and(_offers[fieldNames[0]] == fieldValues[0], _offers[fieldNames[1]] == fieldValues[1])

            for k, v in res.iteritems():
                _offers['_'.join(list(fieldNames) + [k])][ind] = v


def write_tripleDict_to_offers(_offers, _tripleFieldDict):
    partialNames = _tripleFieldDict.values()[0].keys()
    fieldName = '_'.join(['category', 'company', 'brand'])

    for partialName in partialNames:
        _offers[fieldName + '_' + partialName] = None

    for fieldValues, res in _tripleFieldDict.iteritems():

        ind = np.logical_and(_offers['category'] == fieldValues[0],
                             _offers['company'] == fieldValues[1],
                             _offers['brand'] == fieldValues[2])

        for k, v in res.iteritems():
            _offers[fieldName + '_' + k][ind] = v


def repeatFreq(vec):
    """
    number of repeats / number of users
    """
    return 1. * sum(vec > 0) / len(vec)


def avgRepeatPerUser(vec):
    """
    average number of repeats per user
    """
    return np.mean(vec)


def avgRepeatPerRepeatUser(vec):
    """
    average number of repeats per user who repeats
    """
    return 1. * sum(vec) / sum(vec > 0)


def numUsers(vec):
    return len(vec)


def numRepeats(vec):
    return sum(vec > 0)


def totalRepeatTrips(vec):
    return sum(vec)


def _process_chunk_pass2(df, defaultRes):
    if df.shape[0] > 0 and sum(df.numUsers) > 0:
        return {'repeatFreq': 1. * sum(df.numRepeats) / sum(df.numUsers),
                'avgRepeatPerUser': 1. * sum(df.totalRepeatTrips) / sum(df.numUsers),
                'avgRepeatPerRepeatUser': 1. * sum(df.totalRepeatTrips) / sum(df.numRepeats)}
    else:
        return defaultRes

def pass1(pass1OutputFname = "/home/jj/code/Kaggle/ValuedShoppers/Data/offers_amended_1.csv"):
    baseDict = {'avgPrice': 0, 'totalQuantity': 0, 'returnRate': 0, 'count': 0, 'avgReturnAmt': 0}
    singleFieldDict = {name:
                           {v: baseDict.copy() for v in offers[name].unique()}
                       for name in ['category', 'company', 'brand']}

    doubleFieldDict = {name: {tuple(v): baseDict.copy()
                              for v in np.unique(zip(offers[name[0]], offers[name[1]]))}
                       for name in [('company', 'brand'), ('company', 'category'), ('brand', 'category')]}

    tripleFieldDict = {tuple(v): baseDict.copy()
                       for v in np.unique(zip(offers.category, offers.company, offers.brand))}
    chunkSize = 100000
    numLinesLeft = TRANSACTIONS_FILE_NUM_LINES

    # figure out prices
    offers['offerPrice'] = offers.offervalue / offers.quantity

    # fill the dicts
    while numLinesLeft >= chunkSize:
        print numLinesLeft, 'lines left.'
        inner_loop(chunkSize, singleFieldDict, doubleFieldDict, tripleFieldDict)
        numLinesLeft -= chunkSize

    if numLinesLeft > 0:
        inner_loop(numLinesLeft, singleFieldDict, doubleFieldDict, tripleFieldDict)    # the leftover chunk

    transactionsFile.close()

    # post process the dicts
    for curDict in [singleFieldDict, doubleFieldDict]:
        for fieldName, d in curDict.iteritems():
            _postprocess(d)

    _postprocess(tripleFieldDict)

    # write dicts to the offers data frame
    write_singleDict_to_offers(offers, singleFieldDict)
    write_doubleDict_to_offers(offers, doubleFieldDict)
    write_tripleDict_to_offers(offers, tripleFieldDict)

    offers.to_csv(pass1OutputFname, index=False)

def pass2(pass2OutputFname = "/home/jj/code/Kaggle/ValuedShoppers/Data/offers_amended_2.csv"):
    """
    conversion rate by offer. depends only on train history. independent of transaction records.
    """

    defaultDict = {'repeatFreq': np.nan, 'avgRepeatPerUser': np.nan, 'avgRepeatPerRepeatUser': np.nan}
    singleFieldDict_pass2 = {name:
                           {v: defaultDict for v in offers[name].unique()}
                       for name in ['category', 'company', 'brand']}

    doubleFieldDict_pass2 = {name: {tuple(v): defaultDict
                              for v in np.unique(zip(offers[name[0]], offers[name[1]]))}
                       for name in [('company', 'brand'), ('company', 'category'), ('brand', 'category')]}

    tripleFieldDict_pass2 = {tuple(v): defaultDict
                       for v in np.unique(zip(offers.category, offers.company, offers.brand))}


    offersConvDF = pandas.pivot_table(trainHistoryData, rows='offer', values='repeattrips',
                                      aggfunc=[repeatFreq, avgRepeatPerUser, avgRepeatPerRepeatUser, numUsers, numRepeats, totalRepeatTrips]).reset_index()
    mergedDF = pandas.merge(offersConvDF, offers, how='left', on='offer')



    # process single field dict
    for fieldName, d in singleFieldDict_pass2.iteritems():
        for fieldValue in d.keys():
            curDF = mergedDF[mergedDF[fieldName] == fieldValue]
            d[fieldValue] = _process_chunk_pass2(curDF, defaultDict)

    # process double field dict
    for fieldNames, d in doubleFieldDict_pass2.iteritems():
        for fieldValues in d.keys():
            curDF = mergedDF[np.logical_and(mergedDF[fieldNames[0]] == fieldValues[0],
                                            mergedDF[fieldNames[1]] == fieldValues[1])]
            d[fieldValues] = _process_chunk_pass2(curDF, defaultDict)

    # process triple field dict
    fieldNames = ['category', 'company', 'brand']
    for fieldValues in tripleFieldDict_pass2.keys():

        curDF = mergedDF[np.logical_and(mergedDF[fieldNames[0]] == fieldValues[0],
                                           mergedDF[fieldNames[1]] == fieldValues[1],
                                           mergedDF[fieldNames[2]] == fieldValues[2])]
        tripleFieldDict_pass2[fieldValues] = _process_chunk_pass2(curDF, defaultDict)

    # add offer values to singleDict
    newOffers = pandas.merge(offers, offersConvDF, how='left', on='offer')[['offer', 'category', 'company', 'brand'] + defaultDict.keys()]

    # write dicts to the offers data frame
    write_singleDict_to_offers(newOffers, singleFieldDict_pass2)
    write_doubleDict_to_offers(newOffers, doubleFieldDict_pass2)
    write_tripleDict_to_offers(newOffers, tripleFieldDict_pass2)

    newOffers.to_csv(pass2OutputFname, index=False)