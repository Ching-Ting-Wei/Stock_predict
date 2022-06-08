
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
from requests import head
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import math
from sklearn.metrics import classification_report, confusion_matrix


def normalize(data):
    norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return norm


def denormalize(original_data, scaled_data):
    denorm = scaled_data.apply(
        lambda x: x*(np.max(original_data)-np.min(original_data))+np.min(original_data))
    return denorm


def mathew(original, predict):
    tn, fp, fn, tp = confusion_matrix(original, predict).ravel()
    return (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))


def smoothCut(df, days):
    # moving average
    df['close'] = df['close'].rolling(days).mean()
    # drop empty value
    df = df.drop(index=range(days-1), axis=0)
    # calculate diff
    r, c = df.shape
    for i in range(r-1):
        df.iloc[i, 8] = df.iloc[i+1, 7]-df.iloc[i, 7]
    for i in range(r-1):
        if df.iloc[i, 8] > 0:
            df.iloc[i, 10] = 1
        else:
            df.iloc[i, 10] = 0
    #df.to_csv("middle.csv")
    return df


def profit(df, signal):  # df is original input data (from csv file and make close more smooth)
    # y_prediction is predicted data (numpy array)
    # signal is numpy array
    signal = pd.DataFrame(signal)
    signal = signal.reset_index()
    signal = signal.iloc[:, 1]
    df = df.tail(len(signal))
    df = df.reset_index()
    df = df['close']
    current = signal.iloc[0]
    money = 0
    previous = current
    # print(signal)
    # print(df)
    for i in range(1, len(df)):  # i->tomorrow
        if current != signal.iloc[i] and previous == signal.iloc[i]:
            if current == 0:  # buy
                money = money-df.iloc[i-1]  # 扣今天的錢
                current = 1
            else:  # sell
                money = money+df.iloc[i-1]
                current = 0
        previous = signal.iloc[i]  # 前一天是要買還是賣
    return money


money = []
mat = []
mae = []
mse = []
rmse = []
score = pd.DataFrame()
stock_list = [1210, 1231, 2344, 2449, 2603, 2633, 3596, 1215, 1232, 2345, 2454, 2607, 2634, 3682, 1216, 1434, 2379, 2455, 2609,
              2637, 4904, 1218, 1702, 2408, 2459, 2610, 3034, 5388, 1227, 2330, 2412, 2468, 2615, 3035, 1229, 2337, 2439, 2498, 2618, 3045]
for stock in stock_list:
    input_file = "../csv/" + str(stock) + ".csv"
    output_file = "outputr/" + str(stock) + "_result.txt"
    #csv_file = "../csv/pridiction_result/" + str(stock) + ".csv"
    df = pd.read_csv(input_file)
    f = open(output_file, 'w', encoding='utf-8')
    original = df
    df = smoothCut(df, 10)
    HistoricalPrice = df.head(int(len(df)*0.9))
    NewPrice = df.tail(len(df)-len(HistoricalPrice))
    attributes = HistoricalPrice.drop(['X', 'data', 'diff', 'result'], axis=1)
    attributes = attributes.drop('close', axis=1)
    label = HistoricalPrice['close']
    RFRegressor = RandomForestRegressor(
        n_estimators=501, criterion='squared_error', n_jobs=-1)
    print(RFRegressor)
    n_score = cross_val_score(RFRegressor, attributes, label,
                              scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
    print("MSE: %.3f (%.3f)" % (mean(n_score), std(n_score)))
    learned_model = RFRegressor.fit(attributes, label)
    test_attributes = NewPrice.drop('close', axis=1)
    result = test_attributes['result']
    test_attributes = test_attributes.drop(
        ['X', 'data', 'diff', 'result'], axis=1)
    test_lable = NewPrice['close']
    y_prediction = learned_model.predict(test_attributes)
    # print("Mean Absolute Error:",metrics.mean_absolute_error(test_lable,y_prediction))
    # print("Mean Squared Error:",metrics.mean_squared_error(test_lable,y_prediction))
    # print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(test_lable,y_prediction)))
    prediction_result = pd.DataFrame(NewPrice)
    predict_0 = np.array([0])
    for i in range(len(y_prediction)-2):
        if i == 0:
            if y_prediction[i+1] > y_prediction[i]:
                predict_0 = np.array([1])
            else:
                predict_0 = np.array([0])
        if y_prediction[i+1] > y_prediction[i]:
            predict_0 = np.append(predict_0, 1)
        else:
            predict_0 = np.append(predict_0, 0)
    predict_0 = np.append(predict_0, 0)
    money.append(profit(original, predict_0))
    # print(confusion_matrix(result,predict_0))
    # print(classification_report(result,predict_0))
    m = mathew(result, predict_0)
    mat.append(m)
    prediction_result['Prediction_Result'] = y_prediction
    prediction_result['Prediction_Result_0or1'] = predict_0
    output_filename = "outputr/RandomForestR" + str(stock) + ".csv"
    prediction_result.to_csv(output_filename, mode='w',
                             header=True, index=False)

    classResult = classification_report(result,predict_0,output_dict=True)
    score1 = pd.DataFrame(classResult).transpose()
    ndf = score1.unstack().to_frame().T
    ndf.columns = ndf.columns.map('{0[0]}_{0[1]}'.format) 
    score = pd.concat([score, ndf], ignore_index = True, axis = 0)
    mae.append(metrics.mean_absolute_error(test_lable, y_prediction))
    mse.append(metrics.mean_squared_error(test_lable, y_prediction))
    rmse.append(np.sqrt(metrics.mean_squared_error(test_lable, y_prediction)))

    f.write(str(confusion_matrix(result, predict_0)))
    f.write('\n')
    f.write(str(classification_report(result, predict_0)))
    f.write('\n')
    f.write("Mean Absolute Error:")
    f.write(str(metrics.mean_absolute_error(test_lable, y_prediction)))
    f.write('\n')
    f.write("Mean Squared Error:")
    f.write(str(metrics.mean_squared_error(test_lable, y_prediction)))
    f.write('\n')
    f.write("Root Mean Squared Error:")
    f.write(str(np.sqrt(metrics.mean_squared_error(test_lable, y_prediction))))
    f.write('\n')
    f.write("matthews: %.4f" % (m))
d = {'stockID': stock_list, 'profit': money}
p = pd.DataFrame(data=d)
p.to_csv("outputr/profit.csv", header=True, index=False, mode='a')
score.insert(loc=0, column='StockName', value=stock_list)
score['Matthew']=mat
score['MAE']=mae
score['MSE']=mse
score['RMSE']=rmse
score.to_csv("precisionRate_rfr.csv",header = True, index = False,mode='w')