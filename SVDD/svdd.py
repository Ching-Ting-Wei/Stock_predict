
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BaseSVDD import BaseSVDD
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

df = pd.read_csv('2379.csv')
df = df.drop(['date','diff'], axis=1)
CurrentCustomers = df.head(2000)
NewCustomers = df.tail(939)
# print(CurrentCustomers)

label = CurrentCustomers['result']
label = label.to_numpy()
l = len(label)
label = label.reshape(l,1)
attributes = CurrentCustomers.to_numpy()
print(label)
print(attributes)
X_train, X_test, y_train, y_test = train_test_split(attributes, label)
svdd = BaseSVDD(C=0.9, gamma=0.1, kernel='rbf', display='on')

k = 5
scores = cross_val_score(svdd, X_train, y_train, cv=k, scoring='accuracy')

#
print("Cross validation scores:")
for scores_ in scores:
    print(scores_)
 
print("Mean cross validation score: {:4f}".format(scores.mean()))