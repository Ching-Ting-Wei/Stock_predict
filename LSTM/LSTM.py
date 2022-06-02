import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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
df = pd.read_csv(input_directory + filename)
df = df.drop(['data','diff','result'], axis=1)
split_boundary = 2000

train_data = df.head(split_boundary)
test_data_reverse_date = df.tail(df.shape[0] - split_boundary)

#train_data = train_data.drop(['date'], axis=1)
train_data_scaled = normalize(train_data)

X_train = []   #預測點的前 60 天的資料
Y_train = []   #預測點

interval = 30

for i in range(interval, train_data.shape[0]):
    X_train.append(train_data_scaled.iloc[i-interval:i, :-1])
    Y_train.append(train_data_scaled.iloc[i, 5:6])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

test_data = test_data_reverse_date#.drop(["date"], axis=1)
test_data_scaled = normalize(test_data)
X_test = []  
Y_test = np.empty(shape=[0, 1])
backup_test_nparry = np.empty(shape=[0, 1])

for i in range(interval, test_data.shape[0]):
    X_test.append(test_data_scaled.iloc[i-interval:i, :-1])
    Y_test = np.append(Y_test, test_data.iloc[i:i+1, 5:6])
    backup_test_nparry = np.append(backup_test_nparry, test_data_reverse_date.iloc[i:i+1, 0:1])

print(backup_test_nparry.shape)
print(Y_test.shape)
X_test = np.array(X_test)

backup_test = pd.DataFrame({'date': backup_test_nparry, 'real_close': Y_test})
# backup_test = pd.concat([pd.DataFrame({'date': backup_test_nparry}), pd.DataFrame({'real_close': Y_test})], axis=1)
print(backup_test)



# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()


# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))

# # Adding a second LSTM layer and some Dropout regularisation
# regressor.add(LSTM(units = 50, return_sequences = True))
# regressor.add(Dropout(0.2))

# # Adding a third LSTM layer and some Dropout regularisation
# regressor.add(LSTM(units = 50, return_sequences = True))
# regressor.add(Dropout(0.2))

# # Adding a fourth LSTM layer and some Dropout regularisation
# regressor.add(LSTM(units = 50))
# regressor.add(Dropout(0.2))

# # Adding the output layer
regressor.add(Dense(units = 1))

# Compiling
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# 進行訓練
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)

predicted_stock_price = regressor.predict(X_test)
print("Ytest")
print(Y_test)
print("Predicted")
print(predicted_stock_price)

predicted_stock_price = pd.DataFrame(predicted_stock_price)

predicted_stock_price = denormalize(Y_test, predicted_stock_price)  # to get the original scale

backup_test = pd.concat([backup_test, predicted_stock_price], axis=1)

output_directory = os.path.abspath('./output/') 

if not os.path.isdir(output_directory):
  os.makedirs(output_directory)


output_filename = output_directory + "/LSTM_" + str(interval) + '_' + filename

backup_test.to_csv(output_filename, header = True, index = False)

# Visualising the results
# plt.plot(Y_test, color = 'red', label = 'Real Google Stock Price')  # 紅線表示真實股價
# plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')  # 藍線表示預測股價
# plt.title('Google Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Google Stock Price')
# plt.legend()
# plt.show()

# attributes_train, attributes_test, label_train, label_test = train_test_split(attributes, label)
# scaler = StandardScaler()
# scaler.fit(attributes_train)
# nor_attr_train = scaler.transform(attributes_train)
# scaler.fit(attributes_test)
# nor_attr_test = scaler.transform(attributes_test)
# mlp = MLPClassifier(hidden_layer_sizes=(13), max_iter=1500)
# mlp.fit(nor_attr_train,label_train)
# predictions = mlp.predict(nor_attr_test)
# print(confusion_matrix(label_test, predictions))
# print(classification_report(label_test,predictions))