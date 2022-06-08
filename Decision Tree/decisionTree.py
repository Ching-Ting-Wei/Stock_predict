import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn import metrics
def normalize(data):
    norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return norm
import math
def mathew(original, predict):
    tn, fp, fn, tp = confusion_matrix(original,predict).ravel()
    return (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
def smoothCut(df,days):
    #moving average
    df['close']=df['close'].rolling(days).mean()
    #drop empty value
    df=df.drop(index=range(days),axis=0)
    #calculate diff
    r, c = df.shape
    for i in range(r-1):
        df.iloc[i,8] =df.iloc[i+1,7]-df.iloc[i,7]
    for i in range(r-1):
        if df.iloc[i,8]>0:
            df.iloc[i,10]=1
        else:
            df.iloc[i,10]=0
    #df.to_csv("middle.csv")
    return df
def profit(df,signal):#df is original input data (from csv file and make close more smooth)
                                   #y_prediction is predicted data (numpy array)
    #signal is numpy array
    signal = pd.DataFrame(signal)
    signal = signal.reset_index()
    signal = signal.iloc[:,1]
    df = df.tail(len(signal))
    df = df.reset_index()
    df = df['close']
    current = signal.iloc[0]
    money=0
    previous=current
    #print(signal)
    #print(df)
    for i in range(1,len(df)):# i->tomorrow
        if current!=signal.iloc[i] and previous==signal.iloc[i]:
            if current==0:#buy
                money = money-df.iloc[i-1]#扣今天的錢
                current=1
            else:   #sell
                money = money+df.iloc[i-1]
                current=0
        previous=signal.iloc[i]#前一天是要買還是賣
    return money
money = []
stock_list = [1210,1231,2344,2449,2603,2633,3596,1215,1232,2345,2454,2607,2634,3682,1216,1434,2379,2455,2609,2637,4904,1218,1702,2408,2459,2610,3034,5388,1227,2330,2412,2468,2615,3035,1229,2337,2439,2498,2618,3045]
for stock in stock_list:
    print(stock)
    input_file = "../csv/index/" + str(stock) + "_index.csv"
    output_file = "./output/" + str(stock) + "_result.txt"
    # input_file = "../csv/" + str(stock) + ".csv"
    # output_file = "/output/" + str(stock) + "_result.txt"
    csv_file = "./pridiction_result/" + str(stock) + ".csv"
    df = pd.read_csv(input_file)
    original = df
    # df = df.drop(['data','diff', 'X'], axis=1)
    f = open(output_file, 'w',encoding='utf-8')
    # print(df)
    df = smoothCut(df,10)
    CurrentCustomers=df.head(int(len(df)*0.9))
    NewCustomers=df.tail(len(df)-len(CurrentCustomers))
    # CurrentCustomers = df.head(2900)
    # NewCustomers = df.tail(39)
    attributes = CurrentCustomers.drop(['X','data','diff','result'],axis=1)
    label = CurrentCustomers['result']

    DT = DecisionTreeClassifier(criterion='entropy')
    #print(DT)

    scores = cross_val_score(DT, attributes, label, cv=7, scoring='f1_macro',n_jobs=1)
    #print(scores)
    #print("F-score: %0.2f (+/= % 0.2f)" % (scores.mean(),scores.std()*2))

    DT_Model = DT.fit(attributes,label)

    feature_names = attributes.columns[:13]
    # fig = plt.figure(figsize=(50,200))
    # _ = tree.plot_tree(DT_Model, feature_names=feature_names, class_names='result', filled = True)

    test_attributes = NewCustomers.drop(['X','data','diff','result'],axis=1)
    test_label = NewCustomers['result']
    y_prediction = DT_Model.predict(test_attributes)
    money.append(profit(original,y_prediction)) 
    m=mathew(test_label,y_prediction)
    f.write(str(confusion_matrix(test_label,y_prediction)))
    f.write('\n')
    f.write(str(classification_report(test_label,y_prediction)))
    f.write('\n')
    f.write("matthews: %.4f" % (m))
    # print(confusion_matrix(test_label,y_prediction))
    # print(classification_report(test_label,y_prediction))

  #print(confusion_matrix(test_label,y_prediction))
  #print(classification_report(test_label,y_prediction))
    Prediction_result = pd.DataFrame(test_attributes)
    Prediction_result["Prediction_Result"]=y_prediction

    Prediction_result.to_csv(csv_file,mode ='w', header = True)
    #plt.show()
d = {'stockID': stock_list, 'profit': money}
result = pd.DataFrame(data=d)
result.to_csv("output/profit.csv",header = True, index = False,mode='a')
