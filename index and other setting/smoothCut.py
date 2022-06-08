def smoothCut(df,days):
    #moving average
    df['close']=df['close'].rolling(days).mean()
    #drop empty value
    df=df.drop(index=[0,1,2,3,4,5,6,7,8],axis=0)
    #calculate diff
    r, c = df.shape
    for i in range(r-1):
        df.iloc[i,8] =df.iloc[i+1,7]-df.iloc[i,7]
    for i in range(r-1):
        if df.iloc[i,8]>0:
            df.iloc[i,10]=1
        else:
            df.iloc[i,10]=0
    df.to_csv("middle.csv")
    return df