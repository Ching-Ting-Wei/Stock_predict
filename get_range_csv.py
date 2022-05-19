# from sklearn import tree
# from sklearn import datasets
# import pydotplus

file = input("input the stock number: ")

start_year = int(input("input the start year (99~110): "))
end_year = int(input("input the last year (99~110): "))

in_file = "csv/" + file + "_all.csv"
out_file = "csv/" + file + "_" + str(start_year) + "_to_" + str(end_year) + ".csv"

f = open(in_file,'r') # need to modify the name
fout = open(out_file,'w',encoding='utf-8') # need to modify the name

fl = f.readline()
fl = f.readlines()

fout.write("data,volume,price,open,high,low,close,diff,times,result\n")
for line in fl:
	slash = line.find('/')
	y = int(line[0:slash])
	if(y < start_year or y > end_year):
		continue
	fout.write(line)
