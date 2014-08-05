# copied from https://www.kaggle.com/c/liberty-mutual-fire-peril/forums/t/9880/update-on-the-evaluation-metric/51352#post51352

import pandas


def weighted_gini(act,pred,weight):
    df = pandas.DataFrame({"act":act,"pred":pred,"weight":weight})
    df = df.sort('pred',ascending=False)
    df["random"] = (df.weight / df.weight.sum()).cumsum()
    total_pos = (df.act * df.weight).sum()
    df["cum_pos_found"] = (df.act * df.weight).cumsum()
    df["lorentz"] = df.cum_pos_found / total_pos
    n = df.shape[0]
    gini = sum(df.lorentz[1:].values * (df.random[:-1])) - sum(df.lorentz[:-1].values * (df.random[1:]))

    return gini


def normalized_weighted_gini(act,pred, sample_weight):
    return weighted_gini(act,pred,sample_weight) / weighted_gini(act,act,sample_weight)
