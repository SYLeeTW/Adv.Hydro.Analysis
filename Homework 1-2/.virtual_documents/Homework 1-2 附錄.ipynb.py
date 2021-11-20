import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt


def Wilcoxon_RankSum(x1, x2, correction=True):
    n1, n2 = len(x1), len(x2)
    N = n1 + n2
    Rank = pd.concat([x1, x2], ignore_index=False).rank()
    if n1 < n2:
        W = Rank.loc[x1.index[0]].sum()
        mu = n1 * (N+1) / 2
    else:
        W = Rank.loc[x2.index[0]].sum()
        mu = n2 * (N+1) / 2
    
    if correction:
        sigma = np.sqrt(n1*n2 * np.sum(Rank**2) / (N*(N-1)) - 
                        n1*n2 * (N+1)**2 / (4*(N-1)))
    else:
        sigma = np.sqrt(n1 * n2 * (N+1) / 12)
    
    if   W > mu: T = (W - 0.5 - mu) / sigma
    elif W < mu: T = (W + 0.5 - mu) / sigma
    else:        T = 0
    
    print('Statistic T = {:.3f}'.format(T))
    print('Critical Values: +1.96 / -1.96')


Data = pd.read_csv('JulyDMax.csv', index_col='Date')
DMax1 = pd.Series(Data.loc[:,'1961':'1980'].values.flatten(), index=[1]*620)
DMax2 = pd.Series(Data.loc[:,'1990':'2009'].values.flatten(), index=[2]*620)
Wilcoxon_RankSum(DMax1, DMax2)


res = ss.ranksums(DMax2, DMax1)
print('Statistic = {:3.3f}'.format(res.statistic))
print('  p-value = {:.5f}'.format(res.pvalue))


res = ss.mannwhitneyu(DMax2, DMax1)
print('Statistic = {:8.1f}'.format(res.statistic))
print('  p-value = {:.5f}'.format(res.pvalue))


plt.hist(DMax1.values, bins=15)



