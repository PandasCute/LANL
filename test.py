#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/13 14:50
# @Author  : sunjian
# @File    : test.py
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold
np.cumsum([[1,2,3],[2,3,4],[3,4,5]])
files2=os.listdir('./sub2')
data=[]
data2=[]
df3=pd.read_csv('./sub2/'+'1.422.csv',dtype={'seg_id':np.str,'time_to_failure':np.float})
sorted=df3['time_to_failure'].sort_values()
for i in range(len(files2)-1):
    df=df3['time_to_failure']-pd.read_csv('./sub2/'+files2[i],dtype={'seg_id':np.str,'time_to_failure':np.float})['time_to_failure']
    for i in range(len(df)):
        if df[i] > 0:
            df[i] = 1
        else:
            df[i] = -1
    data.append(df)
sum=0
for i in range(len(data)):
    for j in range(i+1,len(data)):
        test=data[i]+data[j]
        sum+=test
sum
sumDF=pd.DataFrame({'sum':sum},index=range(len(sum)))
sumDF.to_csv('./sum1.422.csv',index=False)
for i in range(len(sum)):
    if (-20==sum[i]):
        df3.loc[i,'time_to_failure']=df3.loc[i,'time_to_failure']+(sum[i]/400)
    if (sum[i]==20):
        df3.loc[i,'time_to_failure']=df3.loc[i,'time_to_failure']+(sum[i]/400)
df3.to_csv('./submissions/submission1.csv',index=False)
class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(1)
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# list=os.listdir('./test')
# tests=[]
# for file in list:
#     tests.append(pd.read_csv('./test/'+file,dtype={'acoustic data':np.int,'time_to_failure':np.float}))
# tests
# for df in tests:
#     plt.plot(df['acoustic_data'],color='r')
#     plt.show()