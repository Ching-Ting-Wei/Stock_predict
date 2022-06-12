import pandas as pd
import numpy as np
stock_list = [1210, 1231, 2344, 2449, 2603, 2633, 3596, 1215, 1232, 2345, 2454, 2607, 2634, 3682, 1216, 1434, 2379, 2455, 2609,
              2637, 4904, 1218, 1702, 2408, 2459, 2610, 3034, 5388, 1227, 2330, 2412, 2468, 2615, 3035, 1229, 2337, 2439, 2498, 2618, 3045]
for stock in stock_list:
    algorithm = input('Input the algorithm name: ')
    input_file = "../" + algorithm + "/output/" + \
        algorithm + str(stock) + ".csv"
    output_file = "backtest/" + algorithm + "/" + str(stock) + ".csv"
    df = pd.read_csv("Boost1210.csv")
    borrowsell_mark = []
    borrowbuy_mark = []
    buy_mark = []
    sell_mark = []
    tag1 = 0
    tag2 = 0
    tag3 = 0
    print(len(df))
    for i in range(len(df)-1):
        if tag3 == 1:
            tag3 = 0
            continue
        if df["Prediction_Result"][i] == 1 and df["Prediction_Result"][i+1] == 1:
            if tag1 != 1:
                if tag2 == 1:
                    borrowbuy_mark.append(np.nan)
                    borrowbuy_mark.append(df["close"][i+1])
                    borrowsell_mark.append(np.nan)
                    borrowsell_mark.append(np.nan)
                else:
                    borrowbuy_mark.append(np.nan)
                    borrowbuy_mark.append(np.nan)
                    borrowsell_mark.append(np.nan)
                    borrowsell_mark.append(np.nan)
                sell_mark.append(np.nan)
                sell_mark.append(np.nan)
                buy_mark.append(np.nan)
                buy_mark.append(df["close"][i+1])
                tag1 = 1
                tag2 = 1
                tag3 = 1
            else:
                borrowbuy_mark.append(np.nan)
                borrowsell_mark.append(np.nan)
                sell_mark.append(np.nan)
                buy_mark.append(np.nan)
        elif df["Prediction_Result"][i] == 0 and df["Prediction_Result"][i+1] == 0:
            if tag1 == 1:
                borrowbuy_mark.append(np.nan)
                borrowbuy_mark.append(np.nan)
                borrowsell_mark.append(np.nan)
                borrowsell_mark.append(df["close"][i+1])
                buy_mark.append(np.nan)
                buy_mark.append(np.nan)
                sell_mark.append(np.nan)
                sell_mark.append(df["close"][i+1])
                tag1 = 0
                tag3 = 1
            elif tag1 == 0:
                borrowbuy_mark.append(np.nan)
                borrowbuy_mark.append(np.nan)
                borrowsell_mark.append(np.nan)
                borrowsell_mark.append(df["close"][i+1])
                buy_mark.append(np.nan)
                buy_mark.append(np.nan)
                sell_mark.append(np.nan)
                sell_mark.append(np.nan)
                tag1 = 2
                tag2 = 1
                tag3 = 1
            else:
                borrowbuy_mark.append(np.nan)
                borrowsell_mark.append(np.nan)
                sell_mark.append(np.nan)
                buy_mark.append(np.nan)
        else:
            borrowbuy_mark.append(np.nan)
            borrowsell_mark.append(np.nan)
            sell_mark.append(np.nan)
            buy_mark.append(np.nan)
    if len(buy_mark) != len(df):
        borrowbuy_mark.append(np.nan)
        borrowsell_mark.append(np.nan)
        sell_mark.append(np.nan)
        buy_mark.append(np.nan)
    result = pd.DataFrame(df)
    result.insert(20, column="buy_mark", value=buy_mark)
    result.insert(21, column="sell_mark", value=sell_mark)
    result.insert(22, column="borrowbuy_mark", value=borrowbuy_mark)
    result.insert(23, column="borrowsell_mark", value=borrowsell_mark)
    profit = 0
    for i in range(len(result)):
        profit += sell_mark[i]-buy_mark[i]+borrowsell_mark[i]-borrowbuy_mark[i]
        print(profit)
    print(profit)
#result.to_csv("111.csv", mode='w', header=True)
