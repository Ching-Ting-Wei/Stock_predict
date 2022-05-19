import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
datasets = pd.read_csv('csv/1216.csv')
datasets = datasets.drop(['data','diff'], axis=1)
datasets.shape
print(datasets)

CurrentCustomers = datasets.tail(741)
NewCustomers = datasets.head(250)

attributes = CurrentCustomers.drop('result', axis =1)
label = CurrentCustomers['result']
clf = SVC(kernel='linear', C = 1.0)
scores = cross_val_score = cross_val_score(clf, attributes, label, cv = 10, n_jobs=64)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))

train_attributes = CurrentCustomers.drop('result', axis = 1)
train_label = CurrentCustomers['result']

SVMclassifier = SVC(kernel= 'linear', C= 1.0, random_state=1)
SVMclassifier.fit(train_attributes, train_label)

test_attributes = NewCustomers.drop('result', axis = 1)
test_label = NewCustomers['result']

y_prediction = SVMclassifier.predict(test_attributes)
print(confusion_matrix(test_label, y_prediction))
print(classification_report(test_label, y_prediction))
prediction = pd.DataFrame(test_attributes)
prediction["result"] = y_prediction

prediction.to_csv('prediction_Result_stock.csv', mode ='w', header= True)