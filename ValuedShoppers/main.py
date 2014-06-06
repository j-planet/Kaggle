import pandas
import numpy as np

trainHistory = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/trainHistory_wDateFields.csv")
transactions = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/transactions_small.csv")
offers = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/offers.csv")

# join train history and offers
pandas.merge(trainHistory, offers, left_on='offer', right_on='offer', how='left')

# TODO: compress transaction history. most frequent for each field? count for each field? histogram of each field?
# TODO: convert productmeasure


def compress_transactions():
    pass

