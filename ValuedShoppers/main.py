import pandas
import numpy as np
import matplotlib.pyplot as plt
from dateutil import parser

from sklearn.ensemble import GradientBoostingClassifier

trainHistory = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/trainHistory_wDateFields.csv")
transactions = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/transactions_small.csv")
offers = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/offers.csv")

# join train history and offers
historyAndOffers_train = pandas.merge(trainHistory, offers, left_on='offer', right_on='offer', how='left')
X_train = historyAndOffers_train[['chain', 'category', 'market']]
Y = historyAndOffers_train['repeater']
clf = GradientBoostingClassifier(learning_rate=0.05, loss='deviance', n_estimators=100, subsample=0.9)
clf.fit(X_train, Y)

temp = pandas.DataFrame({'repeattrips': historyAndOffers_train['repeattrips'],
                          'repeater': historyAndOffers_train['repeater'],
                          'pred': clf.predict(X_train),
                          'probs': clf.predict_proba(X_train)[:,1]})
temp.to_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/temp.csv")
# predict
testHistory = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/testHistory_wDateFields.csv")
historyAndOffers_test = pandas.merge(testHistory, offers, left_on='offer', right_on='offer', how='left')
X_test = historyAndOffers_test[['chain', 'category', 'market']]
pandas.DataFrame({'id':testHistory.id, 'repeatProbability': clf.predict(X_test)}).to_csv('/home/jj/code/Kaggle/ValuedShoppers/submissions/initialSub.csv')

# add dates to transactions
dates = [parser.parse(d) for d in transactions.date]  # convert to datetime objects

dayOfTheWeek = [d.weekday() for d in dates]
isWeekend = (np.array(dayOfTheWeek) >= 5)                         # Saturday=5; Sunday=6
days = [d.day for d in dates]
months = [d.month for d in dates]

transactions['dayOfTheWeek'] = dayOfTheWeek
transactions['isWeekend'] = isWeekend
transactions['days'] = days
transactions['months'] = months

del transactions['date']

# TODO: compress transaction history. most frequent for each field? count for each field? histogram of each field?
# TODO: convert productmeasure

# t = transactions[transactions.id == 86246]
t = transactions

for fieldName in t.columns:
    if fieldName in ['id', 'productmeasure']:
        continue
    plt.figure()
    print '--------', fieldName
    _, bins, _ = plt.hist(t[fieldName], bins=30)
    plt.title(fieldName)
plt.show()

temp = np.log(np.abs(t.purchaseamount))

_,bins,_ = plt.hist(temp[temp>0])
plt.show()


def compress_transactions():
    pass

