import numpy as np
import pandas as pd
import os
import sys
#variables
fileName = sys.argv[1]
amount = int(sys.argv[2])
#fileName = "xx.csv"
#amount = 10
#####################
mypath = "data_" + fileName[0:4]
if not os.path.isdir(mypath):
   os.makedirs(mypath)
df = pd.read_csv(fileName)
one_silce = int(len(df)/amount)
for i in range(int(amount/2),amount):
    train = df.head(int(i*one_silce))
    train.to_csv("data_%s/train_%s_%d.csv"%(fileName[0:4],fileName[0:4],i-int(amount/2)+1))
    if i==amount-1:
        test = df.tail(len(df)-i*one_silce)
        print(len(df)-i*one_silce)
    else:
        test = df.iloc[i*one_silce:(i+1)*one_silce]
    test.to_csv("data_%s/test_%s_%d.csv"%(fileName[0:4],fileName[0:4],i-int(amount/2)+1))
