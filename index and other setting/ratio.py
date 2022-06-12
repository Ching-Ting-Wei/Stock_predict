import pandas as pd
import numpy as np

stock_list = [1210, 1231, 2344, 2449, 2603, 2633, 3596, 1215, 1232, 2345, 2454, 2607, 2634, 3682, 1216, 1434, 2379, 2455, 2609,
                    2637, 4904, 1218, 1702, 2408, 2459, 2610, 3034, 5388, 1227, 2330, 2412, 2468, 2615, 3035, 1229, 2337, 2439, 2498, 2618, 3045]

test=0
train=0
total_train=0
total_test=0
for stock in stock_list:
    print(stock)
    input_file = "../csv/index/" + str(stock) + "_index.csv"
    output_file = "output/" + str(stock) + "_result.txt"
    df = pd.read_csv(input_file)
    CurrentCustomers = df.head(int(len(df)*0.9))
    total_train+=len(CurrentCustomers)
    NewCustomers = df.tail(len(df)-len(CurrentCustomers))
    total_test+=len(NewCustomers)
    train+= CurrentCustomers['result'].sum()
    test+= NewCustomers['result'].sum()
print("train ratio: %.3f "%(train/total_train))
print("test ratio: %.3f "%(test/total_test))