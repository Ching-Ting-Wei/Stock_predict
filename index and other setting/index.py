import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
stock_list = [1210,1231,2344,2449,2603,2633,3596,1215,1232,2345,2454,2607,2634,3682,1216,1434,2379,2455,2609,2637,4904,1218,1702,2408,2459,2610,3034,5388,1227,2330,2412,2468,2615,3035,1229,2337,2439,2498,2618,3045]
for stock in stock_list:
    stock_data = pd.read_csv("../csv/"+str(stock)+'.csv')
    # plt.plot(stock_data['close'])
    # plt.show()
    # 計算隔日價差以及分類每日漲跌
    stock_data['diff'] = stock_data['close'].diff()
    stock_data['up'] = stock_data['diff'].clip(lower = 0)
    stock_data['down'] = (-1) * stock_data['diff'].clip( upper = 0)
    # 10日RSI指標
    stock_data['10up'] = stock_data['up'].ewm(com = 10, adjust = False).mean()
    stock_data['10down'] = stock_data['down'].ewm(com = 10, adjust = False).mean()
    stock_data['10relative'] = stock_data['10up'] / stock_data['10down']
    stock_data['rsi10'] = stock_data['10relative'].apply(lambda rs : rs/(1+rs)*100)
    # 5日RSI指標
    stock_data['5up'] = stock_data['up'].ewm(com = 5, adjust = False).mean()
    stock_data['5down'] = stock_data['down'].ewm(com = 5, adjust = False).mean()
    stock_data['5relative'] = stock_data['5up'] / stock_data['5down']
    stock_data['rsi5'] = stock_data['5relative'].apply(lambda rs : rs/(1+rs)*100)

    stock_data=stock_data.drop(['down','up','10up','10down','10relative'],axis=1)
    stock_data=stock_data.drop(['5up','5down','5relative'],axis=1)
    ##macd
    stock_data['12_ema'] = stock_data['close'].ewm(span = 12).mean()
    stock_data['26_ema'] = stock_data['close'].ewm(span = 26).mean()
    stock_data['dif'] = stock_data['12_ema']  - stock_data['26_ema']
    stock_data['macd'] = stock_data['dif'].ewm(span = 9).mean()
    stock_data=stock_data.drop(['12_ema','26_ema','dif'],axis=1)

    ##W%R
    ##5天
    stock_data['5_high'] = stock_data['close'].rolling(5).max()
    stock_data['5_min'] = stock_data['close'].rolling(5).min()
    stock_data['WR5']= (stock_data['close']-stock_data['5_high'])/(stock_data['5_high']-stock_data['5_min'])*100
    stock_data=stock_data.drop(['5_min','5_high'],axis=1)
    ## 10天
    stock_data['5_high'] = stock_data['close'].rolling(10).max()
    stock_data['5_min'] = stock_data['close'].rolling(10).min()
    stock_data['WR10']= (stock_data['close']-stock_data['5_high'])/(stock_data['5_high']-stock_data['5_min'])*100
    stock_data=stock_data.drop(['5_min','5_high'],axis=1)
    stock_data=stock_data.drop(['MinLow','MaxHigh','RSV'],axis=1)
    stock_data=stock_data.fillna(0)
    stock_data.to_csv("../csv/index/"+str(stock)+"_index.csv",header=True,index = False,mode='w')