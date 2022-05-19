import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn import metrics

df = pd.read_csv('2330.csv')
df = df.drop(['data','diff'], axis=1)
print(df)

CurrentCustomers = df.head(741)
NewCustomers = df.tail(250)
attributes = CurrentCustomers.drop('result',axis=1)
label = CurrentCustomers['result']

DT = DecisionTreeClassifier(criterion='entropy')
#print(DT)

scores = cross_val_score(DT, attributes, label, cv=7, scoring='f1_macro',n_jobs=1)
#print(scores)
#print("F-score: %0.2f (+/= % 0.2f)" % (scores.mean(),scores.std()*2))

DT_Model = DT.fit(attributes,label)

feature_names = attributes.columns[:7]
fig = plt.figure(figsize=(50,200))
_ = tree.plot_tree(DT_Model, feature_names=feature_names, class_names='default.payment', filled = True)


test_attributes = NewCustomers.drop('result',axis=1)
test_label = NewCustomers['result']
y_prediction = DT_Model.predict(test_attributes)

print(confusion_matrix(test_label,y_prediction))
print(classification_report(test_label,y_prediction))

Prediction_result = pd.DataFrame(test_attributes)
Prediction_result["Prediction_Result"]=y_prediction

Prediction_result.to_csv('prediction_result_dt.csv',mode ='a', header = True)
#plt.show()
