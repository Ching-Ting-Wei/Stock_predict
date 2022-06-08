import pandas as pd
def profit(df,signal):
    #signal is numpy array
    signal = pd.DataFrame(signal)
    signal = signal.reset_index()
    signal = signal.iloc[:,1]
    df = df.tail(len(signal))
    df = df.reset_index()
    df = df['close']
    current = signal.iloc[0]
    money=0
    previous=current
    print(signal)
    print(df)
    for i in range(1,len(df)):# i->tomorrow
        if current!=signal.iloc[i] and previous==signal.iloc[i]:
            if current==0:#buy
                money = money-df.iloc[i-1]#扣今天的錢
                current=1
            else:   #sell
                money = money+df.iloc[i-1]
                current=0
        previous=signal.iloc[i]#前一天是要買還是賣
    return money