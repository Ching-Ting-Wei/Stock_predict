from numpy import mean
from numpy import std
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

def normalize(data):
  norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
  return norm


filename = input('Input the csv file name: ')
input_directory = os.path.abspath("../csv") + '/'
datasets = pd.read_csv(input_directory + filename)
output_directory = os.path.abspath('./output/') 
if not os.path.isdir(output_directory):
  os.makedirs(output_directory)

CurrentCustomers=datasets.head(2000)
NewCustomers=datasets.tail(939)
NewCustomers.shape

attributes=CurrentCustomers.drop(['data','diff','result'],axis=1)
attributes = normalize(attributes)
label=CurrentCustomers['result']
Boosting_model = AdaBoostClassifier(base_estimator=None,n_estimators=200)
print(Boosting_model)
n_score = cross_val_score(Boosting_model,attributes,label,scoring='f1_macro',cv=10,n_jobs=-1)
print('F-Score: %.3f (%.3f)'%(mean(n_score),std(n_score)))
learned_model=Boosting_model.fit(attributes,label)
test_attributes = NewCustomers.drop(['data','diff','result'],axis=1)
test_label=NewCustomers['result']
test_attributes = normalize(test_attributes)
y_prediction = learned_model.predict(test_attributes)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(test_label,y_prediction))
print(classification_report(test_label,y_prediction))
Predict_result = pd.DataFrame(test_attributes)
Predict_result["Prediction_Result"] = y_prediction
output_filename = output_directory + "/Boosting_" + filename
Predict_result.to_csv(output_filename,mode='a' ,header = True, index = False)
