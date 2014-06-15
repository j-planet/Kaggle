import pandas
import numpy as np
from pprint import pprint

from globalVars import *
from IterStreamer import IterStreamer

offers = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/offers.csv")
transactionsFile = open("/home/jj/code/Kaggle/ValuedShoppers/Data/transactions.csv")
transHeaders = transactionsFile.readline().strip().split(',')
chunkSize = 100000
numLinesLeft = TRANSACTIONS_FILE_NUM_LINES

singleFieldDict = {name: {v: {'avgPrice': 0, 'totalQuantity': 0, 'returnRate': 0, 'count': 0, 'avgReturnAmt': 0}
                          for v in offers[name].unique()}
                   for name in ['category', 'company', 'brand']}
doubleFieldDict = {name: {tuple(v): {'avgPrice': 0, 'totalQuantity': 0, 'returnRate': 0, 'count': 0, 'avgReturnAmt': 0}
                          for v in np.unique(zip(offers[name[0]], offers[name[1]]))}
                   for name in [('company', 'brand'), ('company', 'category'), ('brand', 'category')]}
tripleFieldDict = {tuple(v): {'avgPrice': 0, 'totalQuantity': 0, 'returnRate': 0, 'count': 0, 'avgReturnAmt': 0}
                   for v in np.unique(zip(offers.category, offers.company, offers.brand))}


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


def preprocess_single_dict(chunk, posQuantInd, negQuantInd):

    for fieldName, d in singleFieldDict.iteritems():

        for fieldValue, res in d.iteritems():

            fieldInd = (chunk[fieldName] == fieldValue)
            _process_chunk(chunk, fieldInd, negQuantInd, posQuantInd, res)


def preprocess_double_dict(chunk, posQuantInd, negQuantInd):

    for fieldNames, d in doubleFieldDict.iteritems():

        for fieldValues, res in d.iteritems():

            fieldInd = np.logical_and(chunk[fieldNames[0]] == fieldValues[0], chunk[fieldNames[1]] == fieldValues[1])
            _process_chunk(chunk, fieldInd, negQuantInd, posQuantInd, res)


def preprocess_triple_dict(chunk, posQuantInd, negQuantInd):

    fieldNames = ['category', 'company', 'brand']
    for fieldValues, res in tripleFieldDict.iteritems():

        fieldInd = np.logical_and(chunk[fieldNames[0]] == fieldValues[0], chunk[fieldNames[1]] == fieldValues[1], chunk[fieldNames[2]] == fieldValues[2])
        _process_chunk(chunk, fieldInd, negQuantInd, posQuantInd, res)


def _postprocess(d):
    for fieldValue, res in d.iteritems():
        res['avgPrice'] = res['avgPrice'] / res['count'] if res['count'] > 0 else np.nan
        res['avgReturnAmt'] = res['avgReturnAmt'] / res['returnRate'] if res[
                                                                             'returnRate'] > 0 else np.nan    # returnRate is just a count for now. it's important to do this before the next line
        res['returnRate'] = res['returnRate'] / res['count'] if res['count'] > 0 else np.nan


def postprocess_dicts():

    for curDict in [singleFieldDict, doubleFieldDict]:
        for fieldName, d in curDict.iteritems():
            _postprocess(d)

    _postprocess(tripleFieldDict)


def inner_loop(numLinesToRead):

    rawData = [transactionsFile.next() for _ in xrange(numLinesToRead)]
    chunk = pandas.read_csv(IterStreamer(rawData), names=transHeaders)

    posQuantInd = (chunk.purchasequantity > 0)
    negQuantInd = np.logical_not(posQuantInd)

    preprocess_single_dict(chunk, posQuantInd, negQuantInd)
    preprocess_double_dict(chunk, posQuantInd, negQuantInd)
    preprocess_triple_dict(chunk, posQuantInd, negQuantInd)


def write_singleDict_to_offers():
    partialNames = singleFieldDict.values()[0].values()[0].keys()

    for fieldName, d in singleFieldDict.iteritems():

        for partialName in partialNames:
            offers[fieldName + '_' + partialName] = None

        for fieldValue, res in d.iteritems():
            ind = offers[fieldName] == fieldValue

            for k, v in res.iteritems():
                offers[fieldName + '_' + k][ind] = v


def write_doubleDict_to_offers():
    partialNames = doubleFieldDict.values()[0].values()[0].keys()

    for fieldNames, d in doubleFieldDict.iteritems():

        for partialName in partialNames:
            offers['_'.join(list(fieldNames) + [partialName])] = None

        for fieldValues, res in d.iteritems():

            ind = np.logical_and(offers[fieldNames[0]] == fieldValues[0], offers[fieldNames[1]] == fieldValues[1])

            for k, v in res.iteritems():
                offers['_'.join(list(fieldNames) + [k])][ind] = v


def write_tripleDict_to_offers():
    partialNames = tripleFieldDict.values()[0].keys()
    fieldName = '_'.join(['category', 'company', 'brand'])

    for partialName in partialNames:
        offers[fieldName + '_' + partialName] = None

    for fieldValues, res in tripleFieldDict.iteritems():

        ind = np.logical_and(offers['category'] == fieldValues[0], offers['company'] == fieldValues[1], offers['brand'] == fieldValues[2])

        for k, v in res.iteritems():
            offers[fieldName + '_' + k][ind] = v


# figure out prices
offers['offerPrice'] = offers.offervalue / offers.quantity

# fill the dicts
while numLinesLeft >= chunkSize:
    print numLinesLeft, 'lines left.'
    inner_loop(chunkSize)
    numLinesLeft -= chunkSize

if numLinesLeft > 0:
    inner_loop(numLinesLeft)    # the leftover chunk

transactionsFile.close()

# post process the dicts
postprocess_dicts()

# write dicts to the offers data frame
write_singleDict_to_offers()
write_doubleDict_to_offers()
write_tripleDict_to_offers()

offers.to_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/offers_amended.csv", index=False)
