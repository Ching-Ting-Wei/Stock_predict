## ~~切資料~~
### 輸入方法
```
python3 cut.py [資料名稱] [切幾份]
e.g. python3 cut.py 2303.csv 12
```
資料會放在data_xxx的資料夾內，如果要重新切資料，記得把舊的資料夾刪掉（我懶得寫刪資料了）

# 進度
algorithm|10年(f1 0/1)|5年|3年|
---------|----|---|---|
DecisionTree|*|*|*|
ANN|0.87/0.83|*|*|
SVDD|*|*|*|
LSTM|*|*|*|
Prophet|*|*|*|
RandomForest|*|*|*|
Boosting|*|*|*|
Bagging|*|*|*|

# 資料切法
用KDJ2.py跑過之後
* 10年
  * 訓練資料tail(2000)
  * 測試資料head(939)
