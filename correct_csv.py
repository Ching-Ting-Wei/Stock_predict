stock = input("input the stock name:")
stock_pos = "csv/" + stock + ".csv"
f = open(stock_pos,'r',encoding='utf-8')
# f = open("2454_test2.csv",'r',encoding='utf-8')

lines = f.readline()
lines = f.readlines()
line_cnt = 0
float_set = [4,5,6,7,11,12,13,14,15]
int_set = [2,3,9,10]
error_cnt = 0
for line in lines:
    line_cnt += 1 
    tokens = line.split(',')
    if(len(tokens)!=17):
        print("error tokens number in lines " + str(line_cnt))
        error_cnt += 1
    dates = tokens[1].split('/')
    for date in dates:
        if(not date.isdigit()):
            print("error date in lines " + str(line_cnt))
            error_cnt += 1
    for i in float_set:
        if(not tokens[i].replace('.','').replace('\n','').isdigit()):
            print("error " + str(i) + "th data in lines " + str(line_cnt))
            # print(tokens[i].replace('.',''))
            # print(type(tokens[i].replace('.','')))
            # print(tokens[i].replace('.','').isdigit())
            error_cnt += 1
    for i in int_set:
        if(not tokens[i].isdigit()):
            print("error " + str(i) + "th data in lines " + str(line_cnt))
            error_cnt += 1
    # if(not (tokens[13].replace('\n','') == "1" or tokens[12].replace('\n','') == "0")):
    #     print("error at result in lines " + str(line_cnt))
    #     error_cnt += 1
if(error_cnt == 0):
    print("this csv file is correct")
else:
    print("this csv file has " + str(error_cnt) + "errors.")