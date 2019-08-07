# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
d = pd.read_csv('bill_authentication.csv')
X = d.iloc[:,:-1].values
y =d.iloc[:,-1].values

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

from sklearn.svm import SVC  
#svclassifier = SVC(kernel='linear') 
svclassifier = SVC(C=1, kernel='rbf', gamma=0.01) 
svclassifier.fit(X_train, y_train)  

y_pred = svclassifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix  

cm = confusion_matrix(y_test,y_pred)
acc = ((cm[0][0]+cm[1][1])/cm.sum())*100
print(acc)