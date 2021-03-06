
import math
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt


def normalize(data):
    norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return norm


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

fscore_m = []
fscore_s =[]
money = []
mat = []
score = pd.DataFrame()
stock_list =[1210, 1231, 2344, 2449, 2603, 2633, 3596, 1215, 1232, 2345, 2454, 2607, 2634, 3682, 1216, 1434, 2379, 2455, 2609,
                   2637, 4904, 1218, 1702, 2408, 2459, 2610, 3034, 5388, 1227, 2330, 2412, 2468, 2615, 3035, 1229, 2337, 2439, 2498, 2618, 3045]
for stock in stock_list:
    print(stock)
    input_file = "../csv/index/" + str(stock) + "_index.csv"
    output_file = "output/" + str(stock) + "_result.txt"
    df = pd.read_csv(input_file)
    f = open(output_file, 'w', encoding='utf-8')
    original = df.copy(deep=True)
    df = smoothCut(df, 10)

    CurrentCustomers = df.head(int(len(df)*0.9))
    NewCustomers = df.tail(len(df)-len(CurrentCustomers))
    attributes = CurrentCustomers.drop(['X', 'data', 'diff', 'result'], axis=1)
    attributes = normalize(attributes)
    label = CurrentCustomers['result']
    model = BaggingClassifier(base_estimator=None, n_estimators=500, n_jobs=-1)
    n_score = cross_val_score(
        model, attributes, label, scoring='f1_macro', cv=10, n_jobs=-1, error_score='raise')
    print('F-Score: %.3f (%.3f)' % (mean(n_score), std(n_score)))

    fscore_m.append(mean(n_score))
    fscore_s.append(std(n_score))
    learned_model = model.fit(attributes, label)
    test_attributes = NewCustomers.drop(
        ['X', 'data', 'diff', 'result'], axis=1)
    test_attributes = normalize(test_attributes)
    test_label = NewCustomers['result']
    y_prediction = learned_model.predict(test_attributes)
    from sklearn.metrics import classification_report, confusion_matrix
    m = mathew(test_label, y_prediction)
    mat.append(m)
    money.append(profit(original, y_prediction))
    Predict_result = pd.DataFrame(original.tail(len(test_attributes)))
    Predict_result["Prediction_Result"] = y_prediction
    output_filename = "output/Bagging_" + str(stock) + ".csv"
    Predict_result.to_csv(output_filename, mode='w', header=True, index=False)

    result = classification_report(test_label, y_prediction,output_dict=True)
    score1 = pd.DataFrame(result).transpose()
    ndf = score1.unstack().to_frame().T
    ndf.columns = ndf.columns.map('{0[0]}_{0[1]}'.format) 
    score = pd.concat([score, ndf], ignore_index = True, axis = 0)

    f.write(str(confusion_matrix(test_label, y_prediction)))
    f.write('\n')
    f.write(str(classification_report(test_label, y_prediction)))
    f.write('\n')
    f.write("matthews: %.4f" % (m))
d = {'stockID': stock_list, 'profit': money}
p = pd.DataFrame(data=d)
p.to_csv("output/profit.csv", header=True, index=False, mode='a')
score.insert(loc=0, column='StockName', value=stock_list)
score['Matthew']=mat
score['10F-Score-mean']=fscore_m
score['10F-Score-std']=fscore_s
score.to_csv("precisionRate_bag.csv",header = True, index = False,mode='w')
