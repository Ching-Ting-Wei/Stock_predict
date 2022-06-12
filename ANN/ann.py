
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn import metrics
def normalize(data):
  norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
  return norm

def denormalize(original_data, scaled_data):
  denorm = scaled_data.apply(lambda x: x*(np.max(original_data)-np.min(original_data))+np.min(original_data))
  return denorm
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
fscore_m = []
fscore_s =[]
money = []
mat = []
score = pd.DataFrame()
stock_list = [1210,1231,2344,2449,2603,2633,3596,1215,1232,2345,2454,2607,2634,3682,1216,1434,2379,2455,2609,2637,4904,1218,1702,2408,2459,2610,3034,5388,1227,2330,2412,2468,2615,3035,1229,2337,2439,2498,2618,3045]
for stock in stock_list:
  print(stock)
  # filename = input('Input the csv file name: ')
  # input_directory = os.path.abspath("../csv") + '/'
  # datasets = pd.read_csv(input_directory + filename)
  # output_directory = os.path.abspath('./output/') 
  # if not os.path.isdir(output_directory):
  #   os.makedirs(output_directory)
  input_file = "../csv/index/" + str(stock) + "_index.csv"
  output_file = "./output/" + str(stock) + "_result.txt"
  csv_file = "./pridiction_result/" + str(stock) + ".csv"
  df = pd.read_csv(input_file)
  original = df
  df = smoothCut(df,10)
  f = open(output_file, 'w',encoding='utf-8')
  CurrentCustomers=df.head(int(len(df)*0.9))
  NewCustomers=df.tail(len(df)-len(CurrentCustomers))
  # CurrentCustomers=datasets.head(2000)
  # NewCustomers=datasets.tail(939)

  attributes = CurrentCustomers.drop(['X','data','diff','result','open','high','low'],axis=1)
  label = CurrentCustomers['result']

  attributes_train, attributes_test, label_train, label_test = train_test_split(attributes, label)
  scaler = StandardScaler()
  scaler.fit(attributes_train)
  nor_attr_train = scaler.transform(attributes_train)
  scaler.fit(attributes_test)
  nor_attr_test = scaler.transform(attributes_test)
  from numpy import mean
  from numpy import std
  mlp = MLPClassifier(hidden_layer_sizes=(13), max_iter=15000)
  n_score = cross_val_score(
  mlp, attributes, label, scoring='f1_macro', cv=10, n_jobs=-1, error_score='raise')
#   print('F-Score: %.3f (%.3f)' % (mean(n_score), std(n_score)))

  
  mlp.fit(nor_attr_train,label_train)
  fscore_m.append(mean(n_score))
  fscore_s.append(std(n_score))
  predictions = mlp.predict(nor_attr_test)
  from sklearn.metrics import classification_report, confusion_matrix
  m = mathew(label_test, predictions)
  mat.append(m)
  money.append(profit(original, predictions))

  f.write(str(confusion_matrix(label_test,predictions)))
  f.write('\n')
  f.write(str(classification_report(label_test,predictions)))
  f.write('\n')
  f.write("matthews: %.4f" % (m))
  # print(confusion_matrix(label_test, predictions))
  # print(classification_report(label_test,predictions))
  Prediction_result = pd.DataFrame(attributes_test)
  Prediction_result["Prediction_Result"]=predictions

  Prediction_result.to_csv(csv_file,mode ='w', header = True)
  result = classification_report(label_test, predictions,output_dict=True)
  score1 = pd.DataFrame(result).transpose()
  ndf = score1.unstack().to_frame().T
  ndf.columns = ndf.columns.map('{0[0]}_{0[1]}'.format) 
  score = pd.concat([score, ndf], ignore_index = True, axis = 0)
  #plt.show()
d = {'stockID': stock_list, 'profit': money}
result = pd.DataFrame(data=d)
result.to_csv("output/profit.csv",header = True, index = False,mode='w')
d = {'stockID': stock_list, 'profit': money}
p = pd.DataFrame(data=d)
p.to_csv("output/profit.csv", header=True, index=False, mode='a')
score.insert(loc=0, column='StockName', value=stock_list)
score['Matthew']=mat
score['10F-Score-mean']=fscore_m
score['10F-Score-std']=fscore_s
score.to_csv("precisionRate_bag.csv",header = True, index = False,mode='w')
