fin = open('../csv/index/2449_index.csv','r')
f = open('2449.csv','w',encoding='utf-8')
fir = fin.readline()
f.write(fir)
lines = fin.readlines()
for line in lines:
    if(line.find(",0,") == -1):
        f.write(line)
    else:
        tokens = line.split(',')
        if(tokens[10]=="0"):
            tokens[10] = "-1"
        for i in range(len(tokens)):
            f.write(tokens[i])
            if(i != len(tokens) - 1):
                f.write(",")