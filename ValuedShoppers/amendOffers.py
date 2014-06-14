import pandas
import numpy as np
from globalVars import *
from IterStreamer import IterStreamer

offers = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/offers.csv")
transactionsFile = open("/home/jj/code/Kaggle/ValuedShoppers/Data/transactions.csv")
transHeaders = transactionsFile.readline().strip().split(',')
chunkSize = 10000
numLinesLeft = TRANSACTIONS_FILE_NUM_LINES

singleFieldDict = {name: {v: {'price': 0, 'popularity': 0, 'returnRate': 0, 'count': 0, 'avgReturnAmt': 0}
                          for v in offers[name].unique()}
                   for name in ['category', 'company', 'brand']}
doubleFieldDict = {name: {tuple(v): {'price': 0, 'popularity': 0, 'returnRate': 0, 'count': 0, 'avgReturnAmt': 0}
                          for v in np.unique(zip(offers[name[0]], offers[name[1]]))}
                   for name in [('company', 'brand'), ('company', 'category'), ('brand', 'category')]}
tripleFieldDict = {tuple(v): {'price': 0, 'popularity': 0, 'returnRate': 0, 'count': 0, 'avgReturnAmt': 0}
                   for v in np.unique(zip(offers.category, offers.company, offers.brand))}

# figure out prices
offers['offerPrice'] = offers.offervalue / offers.quantity



while numLinesLeft > 0:

    rawData = [transactionsFile.next() for _ in xrange(chunkSize)]
    chunk = pandas.read_csv(IterStreamer(rawData), names = transHeaders)

    numLinesLeft -= chunkSize

    posQuantInd = (chunk.purchasequantity > 0)
    negQuantInd = np.logical_not(posQuantInd)

    for fieldName, d in singleFieldDict.iteritems():
        for fieldValue, res in d.iteritems():

            fieldInd = (chunk[fieldName] == fieldValue)
            res['count'] += fieldInd.sum()

            # price
            ind = fieldInd & posQuantInd
            res['price'] += 1.*(chunk.purchaseamount[ind] / chunk.purchasequantity[ind]).sum()    # really the total for now

            # popularities
            res['popularity'] += chunk.purchasequantity[fieldInd].sum()

            # number of returns
            res['returnRate'] += (negQuantInd & fieldInd).sum()     # really the count for now
            res['avgReturnAmt'] += chunk.purchaseamount[negQuantInd & fieldInd].sum()   # really the total for now


for fieldName, d in singleFieldDict.iteritems():
    for fieldValue, res in d.iteritems():
        res['price'] /= res['count']
        res['avgReturnAmt'] /= res['returnRate']    # returnRate is just a count for now. it's important to do this before the next line
        res['returnRate'] /= res['count']


offers.to_csv(outputFname)

# if __name__=='__main__':
#     transactions = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/transactions.csv")
#     offers = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/offers.csv")
#
#     amend_offers(offers, transactions, "/home/jj/code/Kaggle/ValuedShoppers/Data/offers_amended.csv")
