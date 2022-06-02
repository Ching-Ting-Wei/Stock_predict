
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
def normalize(data):
  norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
  return norm

def denormalize(original_data, scaled_data):
  denorm = scaled_data.apply(lambda x: x*(np.max(original_data)-np.min(original_data))+np.min(original_data))
  return denorm
  
filename = input('Input the csv file name: ')
input_directory = os.path.abspath("../csv") + '/'
datasets = pd.read_csv(input_directory + filename)
output_directory = os.path.abspath('./output/') 
if not os.path.isdir(output_directory):
  os.makedirs(output_directory)

CurrentCustomers=datasets.head(2000)
NewCustomers=datasets.tail(939)

attributes=CurrentCustomers.drop(['data','diff','result'],axis=1)
label=CurrentCustomers['result']

attributes_train, attributes_test, label_train, label_test = train_test_split(attributes, label)
scaler = StandardScaler()
scaler.fit(attributes_train)
nor_attr_train = scaler.transform(attributes_train)
scaler.fit(attributes_test)
nor_attr_test = scaler.transform(attributes_test)
mlp = MLPClassifier(hidden_layer_sizes=(13), max_iter=1500)
mlp.fit(nor_attr_train,label_train)
predictions = mlp.predict(nor_attr_test)
print(confusion_matrix(label_test, predictions))
print(classification_report(label_test,predictions))