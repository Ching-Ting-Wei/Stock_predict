
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BaseSVDD import BaseSVDD
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

df = pd.read_csv('2454.csv')
df = df.drop(['date','diff'], axis=1)
CurrentCustomers = df.head(2000)
NewCustomers = df.tail(939)
# print(CurrentCustomers)

label = CurrentCustomers['result']
label = label.to_numpy()
l = len(label)
label = label.reshape(l,1)
attributes = CurrentCustomers.to_numpy()
print(label)
print(attributes)
X_train, X_test, y_train, y_test = train_test_split(attributes, label)
# SVDD model
svdd = BaseSVDD(C=0.9, gamma=0.1, kernel='rbf', display='on')
svdd.fit(X_train,  y_train)
y_test_predict = svdd.predict(X_test, y_test)

# plot the distance curve
radius = svdd.radius
distance = svdd.get_distance(X_test)
svdd.plot_distance(radius, distance)

# confusion matrix and ROC curve
cm = confusion_matrix(y_test, y_test_predict)
cm_display = ConfusionMatrixDisplay(cm).plot()
y_score = svdd.decision_function(X_test)

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=3, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=3, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.grid()
plt.show()