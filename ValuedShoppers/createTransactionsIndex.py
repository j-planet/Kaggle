import pandas

f = open("/home/jj/code/Kaggle/ValuedShoppers/Data/transactions.csv")
res = pandas.DataFrame(columns = ['id', 'startRowId', 'endRowId'])

start = 0
lastId = -1

f.readline()        # read header
for i, line in enumerate(f):

    id = line.split(',')[0].strip()

    if not id == lastId:
        res = res.append({'id': lastId, 'startRowId': start, 'endRowId': i-1}, ignore_index=True)
        start = i
        lastId = id

res = res.append({'id': lastId, 'startRowId': start, 'endRowId': i}, ignore_index=True)

f.close()

res.to_csv("/home/jj/code/Kaggle/ValuedShoppers/Data/transIndex.csv", index=False)