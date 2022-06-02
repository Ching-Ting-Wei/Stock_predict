
from numpy import mean
from numpy import std
import os
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np

def normalize(data):
  norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
  return norm
filename = input('Input the csv file name: ')
input_directory = os.path.abspath("../csv") + '/'
datasets = pd.read_csv(input_directory + filename)
output_directory = os.path.abspath('./classOutput/') 
if not os.path.isdir(output_directory):
  os.makedirs(output_directory)

CurrentCustomers=datasets.head(2000)
NewCustomers=datasets.tail(939)
NewCustomers.shape

attributes=CurrentCustomers.drop(['data','diff','result'],axis=1)
label=CurrentCustomers['result']
attributes = normalize(attributes)
RFClassfier = RandomForestClassifier(criterion='gini',n_estimators=500,n_jobs=-1)
print(RFClassfier)
n_score = cross_val_score(RFClassfier,attributes,label,scoring='f1_macro',cv=10,n_jobs=-1)
print('F-Score: %.3f (%.3f)'%(mean(n_score),std(n_score)))

learned_model=RFClassfier.fit(attributes,label)

test_attributes = NewCustomers.drop(['data','diff','result'],axis=1)
test_label=NewCustomers['result']
test_attributes = normalize(test_attributes)
y_prediction = learned_model.predict(test_attributes)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(test_label,y_prediction))
print(classification_report(test_label,y_prediction))
Predict_result = pd.DataFrame(test_attributes)
output_filename = output_directory + "/RandomForestClassfier_" + filename
Predict_result.to_csv(output_filename,mode='a' ,header = True, index = False)