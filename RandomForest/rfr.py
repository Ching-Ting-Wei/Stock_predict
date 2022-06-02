from cgi import test
import os
from xml.dom.minidom import TypeInfo
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
from requests import head
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
def normalize(data):
  norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
  return norm

def denormalize(original_data, scaled_data):
  denorm = scaled_data.apply(lambda x: x*(np.max(original_data)-np.min(original_data))+np.min(original_data))
  return denorm
  
filename = input('Input the csv file name: ')
input_directory = os.path.abspath("../csv") + '/'
datasets = pd.read_csv(input_directory + filename)
output_directory = os.path.abspath('./regressorOutput/') 
if not os.path.isdir(output_directory):
  os.makedirs(output_directory)


HistoricalPrice = datasets.head(2000)
NewPrice = datasets.tail(939)
attributes = HistoricalPrice.drop(['data','diff','result'],axis=1)
attributes = attributes.drop('close',axis=1)
label = HistoricalPrice['close']
RFRegressor = RandomForestRegressor(n_estimators=501, criterion='squared_error',n_jobs=-1)
print(RFRegressor)
n_score = cross_val_score(RFRegressor,attributes,label,scoring='neg_mean_squared_error',cv=10,n_jobs=-1)
print("MSE: %.3f (%.3f)"%(mean(n_score),std(n_score)))
learned_model = RFRegressor.fit(attributes,label)
test_attributes = NewPrice.drop('close',axis=1)
test_attributes = test_attributes.drop(['data','diff','result'],axis=1)
test_lable = NewPrice['close']
y_prediction = learned_model.predict(test_attributes)
print("Mean Absolute Error:",metrics.mean_absolute_error(test_lable,y_prediction))
print("Mean Squared Error:",metrics.mean_squared_error(test_lable,y_prediction))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(test_lable,y_prediction)))
prediction_result = pd.DataFrame(NewPrice)
#print(y_prediction)
prediction_result['Prediction_Result']= y_prediction
output_filename = output_directory + "/RandomForestRegression_" + filename
prediction_result.to_csv(output_filename,mode='w' ,header = True, index = False)
