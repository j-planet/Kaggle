import pandas
from pprint import pprint
import numpy as np
from dateutil import parser
from xlrd.compdoc import x_dump_line
from Kaggle.utilities import RandomForester, print_missing_values_info

from sklearn.ensemble import GradientBoostingClassifier

trainHistory = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/trainHistory_wDateFields.csv")
transactions = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/transactions_train_compressed.csv")
offers = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/offers.csv")

# join train history, offers and transactions
historyOffers = pandas.merge(trainHistory, offers, left_on='offer', right_on='offer', how='left')
historyOffersTrans = pandas.merge(historyOffers, transactions, left_on='id', right_on='id', how='left')


def classify(X, Y):
    clf = GradientBoostingClassifier(learning_rate=0.01, loss='deviance', n_estimators=100, subsample=0.8)
    clf.fit(X, Y)

    temp = pandas.DataFrame({'repeattrips': historyOffers['repeattrips'],
                              'repeater': historyOffers['repeater'],
                              'pred': clf.predict(X),
                              'probs': clf.predict_proba(X)[:,1]})
    temp.to_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/temp.csv", index=False)
    # predict
    testHistory = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/testHistory_wDateFields.csv")
    historyAndOffers_test = pandas.merge(testHistory, offers, left_on='offer', right_on='offer', how='left')
    X_test = historyAndOffers_test[fieldsToUse]
    pandas.DataFrame({'id':testHistory.id, 'repeatProbability': clf.predict_proba(X_test)[:, 1]}).\
            to_csv('/home/jj/code/Kaggle/ValuedShoppers/submissions/initialSub_3.csv', index=False)


def plot_feature_importances(X, Y, labels, numTopFeatures, numEstimators = 50):

    rf = RandomForester(num_features = X.shape[1], n_estimators = numEstimators)
    rf.fit(X, Y)

    print 'Top features:'
    pprint(list(rf.top_indices(labels=labels)[1]))

    rf.plot(num_features=numTopFeatures, labels=labels)




if __name__ == '__main__':
    # fieldsToUse = ['chain', 'category', 'market', 'company', 'brand', 'offervalue']

    fieldsToUse = [f for f in historyOffersTrans.columns if f not in ['id', 'offer', 'repeattrips', 'repeater', 'offerdate']]

    historyOffersTrans_mod = historyOffersTrans[(historyOffersTrans==np.inf).sum(axis=1) == 0]

    X_train = historyOffersTrans_mod[fieldsToUse]

    Y_numRepeats = historyOffersTrans_mod['repeattrips']
    Y_repeater = Y_numRepeats > 0
    assert sum(historyOffersTrans_mod['repeater'][Y_numRepeats > 0]=='f') == 0

    plot_feature_importances(np.array(X_train), Y_repeater, labels=X_train.columns, numTopFeatures=X_train.shape[1]/2)