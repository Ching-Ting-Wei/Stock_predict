import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import os

def normalize(data):
  norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
  return norm

def denormalize(original_data, scaled_data):
  denorm = scaled_data.apply(lambda x: x*(np.max(original_data)-np.min(original_data))+np.min(original_data))
  return denorm

def smoothCut(df,days):
    #moving average
    df['close']=df['close'].rolling(days).mean()
    #drop empty value
    df=df.drop(df.index[range(days-1)],axis=0)
    #calculate diff
    r, c = df.shape
    for i in range(r-1):
        df.iloc[i,8] =df.iloc[i+1,7]-df.iloc[i,7]
    for i in range(r-1):
        if df.iloc[i,8]>0:
            df.iloc[i,10]=1
        else:
            df.iloc[i,10]=0
    # df.to_csv("middle.csv")
    return df


filename = input('Input the csv file name: ')
input_directory = os.path.abspath("../csv/index") + '/'
df = pd.read_csv(input_directory + filename)
df = smoothCut(df, 10)

# df = df.drop(['X', 'diff', 'MinLow', 'MaxHigh'], axis=1)
df = df.drop(['X', 'diff'], axis=1)

# df['close'] = df['close'].rolling(10).mean()
# df = df.drop(df.index[range(9)])


split_boundary = int(df.shape[0] * 0.9)
train_data = df.head(split_boundary)
test_data_reverse_date = df.tail(df.shape[0] - split_boundary)

train_data = train_data.drop(['data', 'result'], axis=1)
train_data_scaled = normalize(train_data)

X_train = []   #預測點的前 60 天的資料
Y_train = []   #預測點

interval = 30

for i in range(interval, train_data.shape[0]):
    X_train.append(train_data_scaled.iloc[i-interval:i, :-1])
    Y_train.append(train_data_scaled.iloc[i, 5:6])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

original_result = pd.DataFrame(test_data_reverse_date['result'])
test_data = test_data_reverse_date.drop(['data', 'result'], axis=1)
test_data_scaled = normalize(test_data)


X_test = []  
real_close = np.empty([0])
original_result_nparry = np.empty([0])
backup_test_nparry = np.empty([0])

for i in range(interval, test_data.shape[0]):
    X_test.append(test_data_scaled.iloc[i-interval:i, :-1])
    real_close = np.append(real_close, test_data.iloc[i:i+1, 5:6])
    backup_test_nparry = np.append(backup_test_nparry, test_data_reverse_date.iloc[i:i+1, 0:1])
    original_result_nparry = np.append(original_result_nparry, original_result.iloc[i:i+1, 0:1])


X_test = np.array(X_test)
# backup_test = pd.concat([pd.DataFrame({'date': backup_test_nparry}), pd.DataFrame({'real_close': Y_test})], axis=1)



# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()


# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 256, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 256, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 256, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 256))
regressor.add(Dropout(0.2))

# # Adding the output layer
regressor.add(Dense(units = 1))

# Compiling
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# 進行訓練
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = pd.DataFrame(predicted_stock_price)

predicted_stock_price = denormalize(real_close, predicted_stock_price)  # to get the original scale

predicted_stock_price_npary = predicted_stock_price.to_numpy()

predicted_stock_price_npary = predicted_stock_price_npary.reshape(-1)

predict_0 = np.empty([0])

for i in range(predicted_stock_price_npary.shape[0]-1):
  if predicted_stock_price_npary[i] < predicted_stock_price_npary[i+1]:
    predict_0 = np.append(predict_0, 1)
  else:
    predict_0 = np.append(predict_0, 0)

predict_0 = np.append(predict_0, 0)
print(backup_test_nparry.shape)
print(real_close.shape)
print(predicted_stock_price_npary.shape)
print(predict_0.shape)

backup_test = pd.DataFrame({'date': backup_test_nparry, 'real_close': real_close, 'predict_close': predicted_stock_price_npary, 'Prediction_Result': predict_0})

output_directory = os.path.abspath('./output/') 

if not os.path.isdir(output_directory):
  os.makedirs(output_directory)

output_filename = output_directory + "/" + str(interval) + '_' + filename
output_report = output_directory + "/" + str(interval) + '_report.txt'

report_file = open(output_report, 'a')

print("=======" + (filename.split("."))[0] + "=======", file=report_file)
print(confusion_matrix(original_result_nparry, predict_0), file=report_file)
print(classification_report(original_result_nparry, predict_0), file=report_file)
print("Mean Absolute Error:",metrics.mean_absolute_error(real_close, predicted_stock_price_npary), file=report_file)
print("Mean Squared Error:",metrics.mean_squared_error(real_close, predicted_stock_price_npary), file=report_file)
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(real_close, predicted_stock_price_npary)), file=report_file)

report_file.close()
backup_test.to_csv(output_filename, header = True, index = False)


# Visualising the results
plt.plot(real_close, color = 'red', label = 'Real Google Stock Price')  # 紅線表示真實股價
plt.plot(predicted_stock_price_npary, color = 'blue', label = 'Predicted Google Stock Price')  # 藍線表示預測股價
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

