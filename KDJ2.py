import matplotlib.pyplot as plt
import pandas as pd

def calKDJ(df):
    df['MinLow'] = df['low'].rolling(9, min_periods=9).min()
    # 填充NaN數據
    df['MinLow'].fillna(value = df['low'].expanding().min(), inplace = True)
    df['MaxHigh'] = df['high'].rolling(9, min_periods=9).max()
    df['MaxHigh'].fillna(value = df['high'].expanding().max(), inplace = True)
    df['RSV'] = (df['close'] - df['MinLow']) / (df['MaxHigh'] - df['MinLow']) * 100
    # 通過for循環依次計算每個交易日的KDJ值
    for i in range(len(df)):
        if i==0:     # 第一天
            df.loc[i,'K']=50
            df.loc[i,'D']=50
        if i>0:
            df.loc[i,'K']=df.loc[i-1,'K']*2/3 + 1/3*df.loc[i,'RSV']
            df.loc[i,'D']=df.loc[i-1,'D']*2/3 + 1/3*df.loc[i,'K']
            df.loc[i,'J']=3*df.loc[i,'D']-2*df.loc[i,'K']
    return df

def drawKDJ():
    stock_list = []
    while True:
        stock = input("please input the stock number: ")
        if stock=="-1":
            break
        stock_list.append(stock)
        print("end of input? input -1")
    
    for stock in stock_list:
        stk_pos = "csv/" + stock + "_all.csv"
        out_pos = "csv/" + stock + "_.csv"
        df = pd.read_csv(stk_pos)
        
        stockDataFrame = calKDJ(df)
        # print(stockDataFrame)
        # 開始繪圖
        # plt.figure()
        # stockDataFrame['K'].plot(color="blue",label='K')
        # stockDataFrame['D'].plot(color="green",label='D')
        stockDataFrame['J'].plot(color="purple",label='J')
        stockDataFrame.to_csv(out_pos,mode ='a', header = True)
        # plt.legend(loc='best')         # 繪製圖例
        # 設置x軸坐標的標籤和旋轉角度    
        # major_index=stockDataFrame.index[stockDataFrame.index%10==0]
        # major_xtics=stockDataFrame['date'][stockDataFrame.index%10==0]
        # plt.xticks(major_index,major_xtics)
        # plt.setp(plt.gca().get_xticklabels(), rotation=30)
        # 帶網格線，且設置了網格樣式
        # plt.grid(linestyle='-.')
        # plt.title("金石資源的KDJ圖")
        # plt.rcParams['font.sans-serif']=['SimHei']
        # plt.show()
# 調用方法
drawKDJ()