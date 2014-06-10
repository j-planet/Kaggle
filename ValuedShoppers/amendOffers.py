import pandas


def amend_offers(offers, transactions, outputFname):

    discountByCategory = []
    discountByBrandNCategory = []
    companyPop = []     # company popularity
    brandPop = []       # brand popularity
    companyReturnRate = []
    brandReturnRate = []

    positiveQuant = (transactions.purchasequantity > 0)

    for rowInd in range(offers.shape[0]):
        print '----- Row', rowInd, '-----'
        row = offers.irow(rowInd)

        # row = offers.irow(0)
        curPrice = row['offervalue'] / row['quantity']

        # discounts
        categoryInd = (transactions.category == row['category']) & positiveQuant
        brandNCategoryInd = (transactions.brand == row['brand']) & categoryInd

        discountByCategory.append(curPrice / (transactions.purchaseamount[categoryInd] / transactions.purchasequantity[categoryInd]).mean())
        discountByBrandNCategory.append(curPrice / (transactions.purchaseamount[brandNCategoryInd] / transactions.purchasequantity[brandNCategoryInd]).mean())

        # popularities
        companyInd = (transactions.company == row['company'])
        brandInd = (transactions.brand == row['brand'])

        companyPop.append((transactions.purchasequantity[companyInd]).sum())
        brandPop.append((transactions.purchasequantity[brandInd]).sum())

        # number of return cases
        companyReturnRate.append(1. * ((transactions.purchaseamount < 0) & companyInd).sum() / companyInd.sum() if companyInd.sum() > 0 else 0)
        brandReturnRate.append(1. * ((transactions.purchaseamount < 0) & brandInd).sum() / brandInd.sum() if brandInd.sum() > 0 else 0)

    offers['discountByCategory'] = discountByCategory
    offers['discountByBrandNCategory'] = discountByBrandNCategory
    offers['companyPop'] = companyPop
    offers['brandPop'] = brandPop
    offers['companyReturnRate'] = companyReturnRate
    offers['brandReturnRate'] = brandReturnRate

    offers.to_csv(outputFname)

if __name__=='__main__':
    transactions = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/transactions.csv")
    offers = pandas.read_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/offers.csv")

    amend_offers(offers, transactions, "/home/jj/code/Kaggle/ValuedShoppers/Data/offers_amended.csv")
