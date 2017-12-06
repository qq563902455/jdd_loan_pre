# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:44:29 2017

@author: 56390
"""

import pandas as pd;
import numpy  as np;
import seaborn as sns;
import math;
import matplotlib.pyplot as plt;
t_click=pd.read_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/t_click.csv",index_col='click_time');
t_user=pd.read_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/t_user.csv",index_col='active_date');
t_loan_sum=pd.read_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/t_loan_sum.csv");
t_loan=pd.read_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/t_loan.csv");
t_order=pd.read_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/t_order.csv");



def getRealValAboutMoney(x):
    if not math.isnan(x):
        return round(5**x-1);
    else:
        return x;

t_user['limit']=t_user['limit'].apply(getRealValAboutMoney);
t_loan_sum['loan_sum']=t_loan_sum['loan_sum'].apply(getRealValAboutMoney);
t_loan['loan_amount']=t_loan['loan_amount'].apply(getRealValAboutMoney);
t_order['price']=t_order['price'].apply(getRealValAboutMoney);
t_order['discount']=t_order['discount'].apply(getRealValAboutMoney);

print(t_click.head());
print(t_user.head());
print(t_loan_sum.head());
print(t_loan.head());
print(t_order.head());


t_click.index=pd.to_datetime(t_click.index,format='%Y-%m-%d',errors='raise');
t_user.index=pd.to_datetime(t_user.index,format='%Y-%m-%d',errors='raise');


print(t_user.describe());
print('由上可知t_user中没有空值');
print('-'*30);


print(t_user.resample('M').count()['uid']);
print('月份个数: ',t_user.resample('M').count().shape[0]);
print('-'*30);

print('初始额度的取值个数: ',t_user.groupby(by=['limit'])['uid'].count().shape[0]);
#plt.figure();
#sns.distplot(t_user['limit']);
print('-'*30);


print('年龄的取值个数: ',t_user.groupby(by=['age'])['uid'].count().shape[0]);
#plt.figure();
#sns.distplot(t_user['age']);
print('-'*30);

print('t_user中性别1的数量',t_user[t_user['sex']==1].shape[0]);
print('t_user中性别2的数量',t_user[t_user['sex']==2].shape[0]);
print('性别1的数量比较多性别1可能是男性')
#plt.figure();
#sns.violinplot(x="age", y="limit",hue='sex', data=t_user,split=True);
print('-'*30);

print('年龄与初始额度的关系:\n        ',t_user.groupby(by=['age'])['limit'].mean());
print('-'*30);

#plt.figure();
#sns.barplot(x="age", y="limit", data=t_user);


t_click.head(3)
print('t_click数据 尺寸：',t_click.shape)
print('t_click中非空值的情况')
print(t_click.count(),' 所以t_click中没有空值的存在')
print('-'*30);
print('t_click中用户uid的个数: ',t_click.groupby(by=['uid'])['uid'].count().shape[0],' 这个数字小于t_user中的数量');
print('t_click中用户pid(点击页面)的个数: ',t_click.groupby(by=['pid'])['pid'].count().shape[0],' 只有10个值,这个是一个cat变量')
#plt.figure();
#sns.distplot(t_click['pid']);
print('t_click中用户param(页面参数)的个数: ',t_click.groupby(by=['param'])['param'].count().shape[0],' 只有48个值,这个是一个cat变量')
#plt.figure();
#sns.distplot(t_click['param'],bins=100);
print('-'*30);
print(t_click.resample('M').count()['uid']);
print('月份个数: ',t_click.resample('M')['uid'].count().shape[0]);
print(t_click.index.max(),'---',t_click.index.min());
print('-'*30);

'''
给数据添加月份这一列
'''
t_click['month']=t_click.index
t_click['month']=t_click['month'].astype(str);
t_click['month']=t_click['month'].apply(lambda x:x[5:7]);
t_click['month']=t_click['month'].astype(int);

tempGroupby=t_click.groupby(by=['uid','month'],as_index=False);
t_user_click=tempGroupby['pid'].count();
t_user_click=t_user_click.rename(columns={'pid':'click_num'});

temp=tempGroupby.apply(lambda x:pd.Series(x['pid']).mode().values);
t_user_click['pid']=temp.values;
temp=tempGroupby.apply(lambda x:pd.Series(x['param']).mode().values);
t_user_click['param']=temp.values;

tempGroupby=t_click.groupby(by=['uid','month','pid'],as_index=False);
temp=tempGroupby['param'].count();
temp=temp.rename(columns={'param':'pid_num'});
for i in range(temp['pid'].min(),temp['pid'].max()+1):
    t_user_click=pd.merge(t_user_click,temp[temp['pid']==i][['uid','month','pid_num']],how='left',on=['uid','month']);
    t_user_click=t_user_click.rename(columns={'pid_num':'pid'+str(i)+'_num'});
    t_user_click['pid'+str(i)+'_ratio']=t_user_click['pid'+str(i)+'_num']/t_user_click['click_num'];
t_user_click=t_user_click.fillna(0);

print('捏一个t_click_user');
print(t_user_click.head());
print('-'*30);


print('t_loan和t_loan_sum中没有空值');
print(t_loan.count())
print(t_loan_sum.count())
t_loan['month']=t_loan['loan_time'].apply(lambda x:x[5:7]).astype(int);
tempGroupby=t_loan.groupby(by=['uid','month'],as_index=False);
t_user_loan=tempGroupby['loan_amount'].sum();
t_user_loan['plannum_amount']=tempGroupby['plannum'].sum()['plannum'];
t_user_loan['loan_average']=tempGroupby['loan_amount'].mean()['loan_amount'];
t_user_loan['plannum_average']=tempGroupby['plannum'].mean()['plannum'];
t_user_loan['burden_per_m']=t_user_loan['loan_amount']/t_user_loan['plannum_amount'];
t_user_loan['loan_num']=tempGroupby['loan_time'].count()['loan_time'];
t_user_loan['loan_max']=tempGroupby['loan_amount'].max()['loan_amount'];
t_user_loan['loan_min']=tempGroupby['loan_amount'].min()['loan_amount'];
t_user_loan.head(20)


print('-'*30);
print('t_order的情况如下:');
print(t_order.head(5));
print('t_order中price中存在空值');
print(t_order.count());
print('price为空时,cate_id的取值');
print(t_order[t_order['price'].isnull()]['cate_id'].unique());
print('时间跨度');
print(t_order['buy_time'].min(),'-',t_order['buy_time'].max());
print('空值补齐策略，采用空值对应cate_id的均值来填补空值');
temp=t_order[['cate_id','price']].dropna().groupby(by=['cate_id'],as_index=False).mean();
t_order=pd.merge(t_order,temp,how='left',on='cate_id');
t_order.loc[t_order['price_x'].isnull(),['price_x']]=t_order[t_order['price_x'].isnull()]['price_y'];
t_order=t_order.drop(['price_y'],axis=1);
t_order=t_order.rename(columns={'price_x':'price'});
print('填充完成，填充后的count结果如下:');
print(t_order.head())
print('-'*30);


print('捏一个t_order_user');
t_order['month']=t_order['buy_time'].apply(lambda x:x[5:7]).astype(int);

t_order['price_sum']=t_order['price']*t_order['qty'];
t_order['actual_pay']=t_order['price_sum']-t_order['discount'];

tempGroupby=t_order.groupby(by=['uid','month'],as_index=False);
t_order_user=tempGroupby['uid','month','price_sum','actual_pay','discount','qty'].sum();

t_order_user['discount_mean']=tempGroupby['discount'].mean()['discount'];
t_order_user['price_sum_mean']=tempGroupby['price_sum'].mean()['price_sum'];
t_order_user['actual_pay_mean']=tempGroupby['actual_pay'].mean()['actual_pay'];

t_order_user['cate_id']=tempGroupby.apply(lambda x:pd.Series(x['cate_id']).mode().values).values;
t_order_user['buy_num']=tempGroupby.count()['price'];
t_order_user['max_price']=tempGroupby['price'].max()['price'];
t_order_user['min_price']=tempGroupby['price'].min()['price'];
t_order_user['average_price']=t_order_user['price_sum']/t_order_user['qty'];
t_order_user['average_pay']=t_order_user['actual_pay']/t_order_user['qty'];
t_order_user['average_discount']=t_order_user['discount']/t_order_user['qty'];

print('捏完后的效果如下:')
print(t_order_user.head())
print('-'*30);
#plt.figure();
#sns.distplot(t_order_user['price'],bins=50);
#
#plt.figure();
#sns.distplot(t_order_user['average_price'],bins=50);
#
#plt.figure();
#sns.distplot(t_order_user['buy_num'],bins=50);
#
#plt.figure();
#sns.distplot(t_order_user['discount'],bins=50);
#
#plt.figure();
#sns.regplot(x="price", y="discount", data=t_order_user);
#
#plt.figure();
#sns.regplot(x="average_price", y="discount", data=t_order_user);


print('开始合并数据集，开始做一些统计分析:')
print('根据观察t_loan_sum里的数据,就是t_user_loan里面11月的数据');

print('合并t_user与t_user_loan');
tempList=[];
for i in range(8,12):
    temp=t_user.copy();
    temp['month']=i;
    tempList.append(temp);
    
t_merge=pd.concat(tempList);

t_merge=pd.merge(left=t_merge,right=t_user_loan,on=['uid','month'],how='left');
t_merge[['plannum_amount','plannum_average']]=t_merge[['plannum_amount','plannum_average']].fillna(1);
t_merge=t_merge.fillna(0);
t_merge=pd.merge(left=t_merge,right=t_order_user,on=['uid','month'],how='left');
t_merge=t_merge.fillna(0);
t_merge=pd.merge(left=t_merge,right=t_user_click,on=['uid','month'],how='left');
t_merge=t_merge.fillna(0);
print(t_merge.head());
print(t_merge.count());

print('合并数据完成');
'''
关于t_merge的信息
uid             用户id
age             用户年龄
sex             用户性别
limit           初始额度
month           月份
loan_amount     该月份贷款总额
plannum         该月平均每次借贷的分期数
loan_num        该月贷款次数
price           该月购买物品的总价格
qty             该月购买物品的总数
discount        该月购买物品的平均折扣
cate_id         物品的类别id
buy_num         购买的次数
average_price   平均每一次购买的价格 
click_num       该月点击的总次数
pid             该月最常去的页面id
param           该月最常去的页面的参数
'''



print('对于这个数据集做一些分析');
print('首先把每个月的数据 横向展开')
t_merge[t_merge['month']==8].head()
t_merge[t_merge['month']==9].head()
t_merge[t_merge['month']==10].head()
t_merge[t_merge['month']==11].head()

t_vis=pd.DataFrame();
excludeCols=['uid','age','sex','limit'];
t_vis=t_merge[t_merge['month']==9][excludeCols];
excludeCols.append('month');

for col in t_merge.columns:
    if col not in excludeCols:
        for i in range(8,12):
            t_vis[col+'_'+str(i)]=t_merge[t_merge['month']==i][col].values;
    
print('新的数据集生成完成，整个数据集中_x，x表示月份');
print(t_vis.head())


#plt.figure();
#sns.regplot(x="loan_amount_10", y="loan_amount_11", data=t_vis);
#plt.figure();
#sns.regplot(x="loan_amount_9", y="loan_amount_11", data=t_vis);
#plt.figure();
#sns.regplot(x="loan_amount_8", y="loan_amount_11", data=t_vis);
#
#plt.figure();
#sns.distplot(t_vis['loan_amount_11'],bins=100);
#
#plt.figure();
#sns.distplot(t_vis['loan_amount_10'],bins=100);
#
#plt.figure();
#sns.distplot(t_vis['loan_amount_9'],bins=100);
#
#plt.figure();
#sns.distplot(t_vis['loan_amount_8'],bins=100);
#
#
#corrmat=t_vis[['loan_amount_11','loan_amount_10','loan_amount_9','loan_amount_8']].corr();
#f, ax = plt.subplots(figsize=(9, 6))
#sns.heatmap(corrmat, vmax=.8, square=True);
def getListFirstVal(val):
    if type(val)==np.ndarray:
        return val[0];
    else:
        return val;
for col in t_vis.columns:
    if 'pid' in col or 'param' in col or 'cate_id' in col:
        t_vis[col]=t_vis[col].apply(getListFirstVal);

print('各种处理都完成后的t_vis表如下：');
print(t_vis.head())

t_vis.to_csv("D:/general_file_unclassified/about_code/JDD/Loan_pre/data/t_vis.csv")
print('end');




#sns.barplot(x="param_9", y="loan_amount_10",data=t_vis);
#sns.barplot(x="pid_10", y="loan_amount_11",data=t_vis);
#sns.barplot(x="pid_11", y="loan_amount_11",data=t_vis);






