# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 18:47:17 2017

@author: 56390
"""
import pandas as pd;
import numpy  as np;
import seaborn as sns;
import matplotlib.pyplot as plt;

import math;


t_vis=pd.read_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/t_vis.csv");
t_vis=t_vis.drop(['Unnamed: 0'],axis=1);
ans=pd.read_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/submit.csv");
#ans['loan_amount']=5**ans['loan_amount']-1;


def getEncryptVal(val):
    if val<0:
        return 0;
    else:
        return math.log(val+1,5);


t_vis['loan_amount_11'].apply(getEncryptVal).mean()
t_vis['loan_amount_10'].apply(getEncryptVal).mean()
t_vis['loan_amount_9'].apply(getEncryptVal).mean()
t_vis['loan_amount_8'].apply(getEncryptVal).mean()


temp_vis=t_vis[['uid','loan_amount_8','loan_amount_9','loan_amount_10','loan_amount_11']];

temp_vis.max()

t_vis

ans.mean()

t_vis.count()





corrmat = t_vis.corr()
f, ax = plt.subplots(figsize=(36, 24))
sns.heatmap(corrmat, vmax=.8, square=True);
sns.regplot(x="burden_per_m_10", y="loan_amount_11", data=t_vis);



t_vis.shape

t_user=pd.read_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/t_user.csv");


len(t_user['uid'].unique())

t_order=pd.read_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/t_order.csv");
t_order['price']=5**t_order['price']-1;
t_order['discount']=5**t_order['discount']-1;


temp=t_order[t_order['uid']==11801][t_order['buy_time']!='2016-11-07'];

t_order.count()


5**t_order[t_order['discount']!=0].head()['discount']-1

t_order.head()



