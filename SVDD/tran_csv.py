fin = open('../csv/2379.csv','r')
f = open('2379.csv','w',encoding='utf-8')
fir = fin.readline()
f.write(fir)
lines = fin.readlines()
for line in lines:
    if(line[-2]=="0"):
        f.write(line[:-3])
        f.write(",-1\n")
    else:
        f.write(line)