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


t_loan=pd.read_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/t_loan.csv");


t_loan['loan_amount']=5**t_loan['loan_amount']-1;


t_loan['month']=t_loan['loan_time'].apply(lambda x:x[5:7]).astype(int);
t_loan['hour']=t_loan['loan_time'].apply(lambda x:str(x)[11:13]).astype(int);

tempGroupby=t_loan.groupby(by=['uid','month'],as_index=False);
t_user_loan=tempGroupby['loan_amount'].sum();
t_user_loan['plannum_amount']=tempGroupby['plannum'].sum()['plannum'];
t_user_loan['loan_average']=tempGroupby['loan_amount'].mean()['loan_amount'];
t_user_loan['plannum_average']=tempGroupby['plannum'].mean()['plannum'];
t_user_loan['burden_per_m']=t_user_loan['loan_amount']/t_user_loan['plannum_amount'];
t_user_loan['loan_num']=tempGroupby['loan_time'].count()['loan_time'];
t_user_loan['loan_max']=tempGroupby['loan_amount'].max()['loan_amount'];
t_user_loan['loan_min']=tempGroupby['loan_amount'].min()['loan_amount'];
t_user_loan.head(5)


tempGroupby=t_loan.groupby(by=['uid','month','hour'],as_index=False);
temp=tempGroupby['loan_amount','plannum'].sum();
temp['loan_num']=tempGroupby['loan_amount'].count()['loan_amount'];


for i in range(0,24):
    rawNameList=temp.columns;
    temp=temp.rename(columns={'loan_amount':'loan_amount_'+str(i)+'hour','plannum':'plannum_'+str(i)+'hour','loan_num':'loan_num_'+str(i)+'hour'});    
    t_user_loan=pd.merge(t_user_loan,temp[temp['hour']==i].drop(['hour'],axis=1),how='left',on=['uid','month']);
    temp.columns=rawNameList;
t_user_loan=t_user_loan.fillna(0);

for i in range(0,24):
    t_user_loan['loan_amount_'+str(i)+'hour'+'_ratio']=t_user_loan['loan_amount_'+str(i)+'hour']/t_user_loan['loan_amount'];
t_user_loan['mostly_loan_hour']=0;


hourList=[];
for i in range(0,24):
    hourList.append('loan_amount_'+str(i)+'hour_ratio');
t_user_loan['loan_amount_hour_max_ratio']=t_user_loan[hourList].max(axis=1);

for i in range(0,24):
     t_user_loan.loc[t_user_loan['loan_amount_'+str(i)+'hour_ratio']==t_user_loan['loan_amount_hour_max_ratio'],'mostly_loan_hour']=i;



nightList=[];
for i in [23,0,1,2,3,4,5,6]:
    nightList.append('loan_amount_'+str(i)+'hour_ratio');
dayList=[];
for i in range(0,24):
    if i not in [23,0,1,2,3,4,5,6]:
        dayList.append('loan_amount_'+str(i)+'hour_ratio');


t_user_loan['loan_amount_day_ratio']=t_user_loan[dayList].sum(axis=1);
t_user_loan['loan_amount_night_ratio']=t_user_loan[nightList].sum(axis=1);

t_user_loan['mostly_loan_day_or_night']=1;
t_user_loan.loc[t_user_loan['loan_amount_day_ratio']<0.5,'mostly_loan_day_or_night']=0;



sns.regplot(x="loan_amount_night_ratio", y="loan_amount", data=t_user_loan);
sns.barplot(x="mostly_loan_day_or_night", y="loan_amount",data=t_user_loan[t_user_loan['month']==10]);

























t_loan['loan_amount']=5**t_loan['loan_amount']-1;
t_loan.index=pd.to_datetime(t_loan.index,format='%Y-%m-%d',errors='raise');
t_loan['hour']=t_loan.index;
t_loan['hour']=t_loan['hour'].apply(lambda x:str(x)[11:13]).astype(int);
t_loan['daytime']=1;
t_loan.loc[t_loan['hour']<7,'daytime']=0;
t_loan.loc[t_loan['hour']>22,'daytime']=0;
t_loan['month']=t_loan.index;
t_loan['month']=t_loan['month'].apply(lambda x:str(x)[5:7]).astype(int);



t_loan.groupby(by=['daytime','month'],as_index=False).sum().head()

#(5**t_loan[t_loan['loan_time']>'2016-9'][t_loan['loan_time']<'2016-10'][t_loan['loan_time']<'2016-10-15']['loan_amount']-1).sum()
#(5**t_loan[t_loan['loan_time']>'2016-9'][t_loan['loan_time']<'2016-10'][t_loan['loan_time']>'2016-10-15']['loan_amount']-1).sum()


tempGroupby=t_loan.groupby(by=['uid','daytime','month'],as_index=False);
tempSet=tempGroupby['uid','daytime','month','loan_amount','plannum'].sum();


tempSet.head()

#t_loan_day.head()

sns.barplot(x="month", y="loan_amount",hue='daytime', data=t_loan_day);


t_loan.columns
ans=pd.read_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/submit.csv");



t_loan_day[t_loan_day['loan_time']=='2016'].head(0)

t_loan_day['loan_time'].head()

t_loan_day['loan_time'].apply(lambda x:str(x)[11:13]).astype(int).unique()











t_order=pd.read_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/t_order.csv",index_col='buy_time');

t_order['price']=5**t_order['price']-1;
t_order['discount']=5**t_order['discount']-1;


t_order.index=pd.to_datetime(t_order.index,format='%Y-%m-%d',errors='raise');
t_order_day=t_order.resample('D').sum();
t_order_day['buy_time']=t_order_day.index;

t_order_day.head()

sns.barplot(x="buy_time", y="price", data=t_order_day['2016-11']);


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



