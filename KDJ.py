import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# df = pd.read_csv('csv/2454.csv')
# low_list = df["close"].rolling(9, min_periods=1).min()
# high_list = df["high"].rolling(9, min_periods=1).max()
# rsv = (df["close"] - low_list) / (high_list - low_list) * 100
# df["K"] = rsv.ewm(com=2, adjust=False).mean()
# df["D"] = df["K"].ewm(com=2, adjust=False).mean()
# df["J"] = 3 * df["K"] - 2 * df["D"]
# pd.DataFrame.ewm(self, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, axis=0)

# plt.figure(figsize=(12, 8))
plt.figure(figsize=(60, 8))
df = pd.read_csv('csv/2454.csv')
low_list = df["close"].rolling(9, min_periods=1).min()
high_list = df["high"].rolling(9, min_periods=1).max()
rsv = (df["close"] - low_list) / (high_list - low_list) * 100
df["K"] = rsv.ewm(com=2, adjust=False).mean()
df["D"] = df["K"].ewm(com=2, adjust=False).mean()
df["J"] = 3 * df["D"] - 2 * df["K"]
df.to_csv('2454_test.csv',mode ='a', header = True)
plt.plot(df["date"], df["K"], label ="K")
plt.plot(df["date"], df["D"], label ="D")
plt.plot(df["date"], df["J"], label ="J")
plt.legend()
plt.show()
