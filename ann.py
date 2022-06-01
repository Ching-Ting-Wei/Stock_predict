
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('csv/2454.csv')
df = df.drop(['date','diff'], axis=1)
CurrentCustomers = df.head(2000)
NewCustomers = df.tail(939)

attributes = CurrentCustomers.drop('result',axis=1)
label = CurrentCustomers['result']

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