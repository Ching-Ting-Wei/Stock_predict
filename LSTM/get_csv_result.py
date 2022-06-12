fin = open('./output/30_report.txt')
f = open('percisionRate_LSTM', 'w', encoding='utf-8')

lines = fin.readlines()
f.write("StockName,precision_0,precision_1,precision_accuracy,precision_macro avg,precision_weighted avg,recall_0,recall_1,recall_accuracy,recall_macro avg,recall_weighted avg,f1-score_0,f1-score_1,f1-score_accuracy,f1-score_macro avg,f1-score_weighted avg,support_0,support_1,support_accuracy,support_macro avg,support_weighted avg,MAE,MSE,RMSE\n")
for i in range(40):
    f.write(lines[i*15][7:11] + ',')
    
    f.write(lines[i*15 + 5][19:23] + ',')
    f.write(lines[i*15 + 6][19:23] + ',')
    f.write(lines[i*15 + 8][39:43] + ',')
    f.write(lines[i*15 + 9][19:23] + ',')
    f.write(lines[i*15 + 10][19:23] + ',')

    f.write(lines[i*15 + 5][29:33] + ',')
    f.write(lines[i*15 + 6][29:33] + ',')
    f.write(lines[i*15 + 8][39:43] + ',')
    f.write(lines[i*15 + 9][29:33] + ',')
    f.write(lines[i*15 + 10][29:33] + ',')
    
    f.write(lines[i*15 + 5][39:43] + ',')
    f.write(lines[i*15 + 6][39:43] + ',')
    f.write(lines[i*15 + 8][39:43] + ',')
    f.write(lines[i*15 + 9][39:43] + ',')
    f.write(lines[i*15 + 10][39:43] + ',')
    
    f.write(lines[i*15 + 5][50:53] + ',')
    f.write(lines[i*15 + 6][50:53] + ',')
    f.write(lines[i*15 + 8][50:53] + ',')
    f.write(lines[i*15 + 9][50:53] + ',')
    f.write(lines[i*15 + 10][50:53] + ',')
    
    f.write(lines[i*15 + 12][21:-1] + ',')
    f.write(lines[i*15 + 13][20:-1] + ',')
    f.write(lines[i*15 + 14][25:] )