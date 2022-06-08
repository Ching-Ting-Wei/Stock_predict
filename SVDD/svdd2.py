
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BaseSVDD import BaseSVDD
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
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
df = pd.read_csv('2449.csv')
df = df.drop(['data','diff','X'], axis=1)
df = smoothCut(df,10)
CurrentCustomers=df.head(int(len(df)*0.9))
NewCustomers=df.tail(len(df)-len(CurrentCustomers))
CurrentCustomers = df.head(2000)
NewCustomers = df.tail(939)
# print(CurrentCustomers)

label = CurrentCustomers['result']
print(label)
label = label.to_numpy()
l = len(label)
label = label.reshape(l,1)
attributes = CurrentCustomers.to_numpy()
# print(label)
# print(attributes)
X_train, X_test, y_train, y_test = train_test_split(attributes, label)
# SVDD model
svdd = BaseSVDD(C=0.9, gamma=0.1, kernel='rbf', display='on')
svdd.fit(X_train,  y_train)
y_test_predict = svdd.predict(X_test, y_test)

# plot the distance curve
radius = svdd.radius
distance = svdd.get_distance(X_test)
# svdd.plot_distance(radius, distance)

# confusion matrix and ROC curve
cm = confusion_matrix(y_test, y_test_predict)
cm_display = ConfusionMatrixDisplay(cm).plot()
y_score = svdd.decision_function(X_test)

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# plt.figure()
# plt.plot(fpr, tpr, color="darkorange", lw=3, label="ROC curve (area = %0.2f)" % roc_auc)
# plt.plot([0, 1], [0, 1], color="navy", lw=3, linestyle="--")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver operating characteristic")
# plt.legend(loc="lower right")
# plt.grid()
# plt.show()